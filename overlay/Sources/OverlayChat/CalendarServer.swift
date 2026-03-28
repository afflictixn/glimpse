import Foundation
import Contacts
import EventKit
import Network
import SQLite3

/// Local HTTP server (port 9322) that exposes macOS-protected data to the Python backend.
/// Routes:
///   GET /calendar?days=7     — calendar events (requires Calendar permission)
///   GET /contacts/search?q=John  — search contacts by name, email, or phone (requires Contacts permission)
///   GET /imessage/recent?limit=10  — recent iMessage conversations (requires Full Disk Access)
///   GET /imessage/search?q=hello&limit=20  — search iMessage history
///   GET /imessage/conversation?contact=John&limit=30  — messages with a contact
final class CalendarServer {
    private let store = EKEventStore()
    private let contactStore = CNContactStore()
    private var listener: NWListener?
    private let queue = DispatchQueue(label: "overlay-server", qos: .utility)
    private(set) var hasCalendarAccess = false
    private(set) var hasContactsAccess = false

    private let chatDBPath: String = {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return "\(home)/Library/Messages/chat.db"
    }()

    private let mailDBPath: String = {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let base = "\(home)/Library/Mail"
        // Find the latest V* directory
        let fm = FileManager.default
        if let contents = try? fm.contentsOfDirectory(atPath: base) {
            let versions = contents.filter { $0.hasPrefix("V") }.sorted().reversed()
            for v in versions {
                let path = "\(base)/\(v)/MailData/Envelope Index"
                if fm.fileExists(atPath: path) { return path }
            }
        }
        return "\(base)/V10/MailData/Envelope Index"
    }()

    func start() {
        requestCalendarAccess()
        requestContactsAccess()
        startListener()
    }

    func stop() {
        listener?.cancel()
    }

    // MARK: - Calendar Permission

    private func requestCalendarAccess() {
        let status = EKEventStore.authorizationStatus(for: .event)

        if #available(macOS 14.0, *) {
            if status == .fullAccess {
                hasCalendarAccess = true
                print("[server] calendar: already authorized (fullAccess)")
                return
            }
        }
        if status == .authorized {
            hasCalendarAccess = true
            print("[server] calendar: already authorized")
            return
        }

        if #available(macOS 14.0, *) {
            store.requestFullAccessToEvents { [weak self] granted, error in
                self?.hasCalendarAccess = granted
                print("[server] calendar fullAccess: granted=\(granted)")
            }
        } else {
            store.requestAccess(to: .event) { [weak self] granted, error in
                self?.hasCalendarAccess = granted
                print("[server] calendar access: granted=\(granted)")
            }
        }
    }

    // MARK: - Contacts Permission

    private func requestContactsAccess() {
        let status = CNContactStore.authorizationStatus(for: .contacts)
        if status == .authorized {
            hasContactsAccess = true
            print("[server] contacts: already authorized")
            return
        }
        contactStore.requestAccess(for: .contacts) { [weak self] granted, error in
            self?.hasContactsAccess = granted
            print("[server] contacts access: granted=\(granted)")
        }
    }

    // MARK: - HTTP Listener

    private func startListener() {
        let params = NWParameters.tcp
        listener = try? NWListener(using: params, on: 9322)

        listener?.stateUpdateHandler = { state in
            switch state {
            case .ready:
                print("[server] listening on port 9322 (calendar + imessage)")
            case .failed(let err):
                print("[server] listener failed: \(err)")
            default:
                break
            }
        }

        listener?.newConnectionHandler = { [weak self] conn in
            self?.handleConnection(conn)
        }

        listener?.start(queue: queue)
    }

    // MARK: - Request Routing

    private func handleConnection(_ connection: NWConnection) {
        connection.start(queue: queue)
        connection.receive(minimumIncompleteLength: 1, maximumLength: 8192) { [weak self] data, _, _, error in
            guard let self = self, let data = data else {
                connection.cancel()
                return
            }

            let raw = String(data: data, encoding: .utf8) ?? ""
            let response: String

            if raw.contains("/contacts/search") {
                response = self.handleContactsSearch(raw)
            } else if raw.contains("/imessage/recent") {
                response = self.handleIMessageRecent(raw)
            } else if raw.contains("/imessage/search") {
                response = self.handleIMessageSearch(raw)
            } else if raw.contains("/imessage/conversation") {
                response = self.handleIMessageConversation(raw)
            } else if raw.contains("/mail/recent") {
                response = self.handleMailRecent(raw)
            } else if raw.contains("/mail/search") {
                response = self.handleMailSearch(raw)
            } else if raw.contains("/calendar") {
                response = self.handleCalendar(raw)
            } else {
                response = "{\"error\": \"Unknown endpoint.\"}"
            }

            let httpResponse = self.buildHTTPResponse(body: response)
            connection.send(content: httpResponse.data(using: .utf8), completion: .contentProcessed { _ in
                connection.cancel()
            })
        }
    }

    // MARK: - Calendar Handler

    private func handleCalendar(_ raw: String) -> String {
        let days = min(parseIntParam(raw, name: "days", fallback: 7), 30)

        guard hasCalendarAccess else {
            return "{\"error\": \"Calendar access not granted. Allow in System Settings > Privacy > Calendars.\"}"
        }

        let start = Date()
        let end = Calendar.current.date(byAdding: .day, value: days, to: start)!
        let predicate = store.predicateForEvents(withStart: start, end: end, calendars: nil)
        let events = store.events(matching: predicate)

        if events.isEmpty {
            return "{\"events\": [], \"note\": \"No events in the next \(days) days.\"}"
        }

        let fmt = ISO8601DateFormatter()
        var results: [[String: Any]] = []
        for event in events.prefix(50) {
            var entry: [String: Any] = [
                "title": event.title ?? "",
                "start": fmt.string(from: event.startDate),
                "end": fmt.string(from: event.endDate),
                "all_day": event.isAllDay,
            ]
            if let loc = event.location, !loc.isEmpty { entry["location"] = loc }
            if let notes = event.notes, !notes.isEmpty { entry["notes"] = String(notes.prefix(200)) }
            results.append(entry)
        }

        return toJSON(results)
    }

    // MARK: - Contacts Handler

    private func handleContactsSearch(_ raw: String) -> String {
        let query = parseStringParam(raw, name: "q") ?? ""
        guard !query.isEmpty else {
            return "{\"error\": \"Missing required parameter: q\"}"
        }
        guard hasContactsAccess else {
            return "{\"error\": \"Contacts access not granted. Allow in System Settings > Privacy > Contacts.\"}"
        }

        let keys: [CNKeyDescriptor] = [
            CNContactGivenNameKey as CNKeyDescriptor,
            CNContactFamilyNameKey as CNKeyDescriptor,
            CNContactEmailAddressesKey as CNKeyDescriptor,
            CNContactPhoneNumbersKey as CNKeyDescriptor,
            CNContactBirthdayKey as CNKeyDescriptor,
            CNContactOrganizationNameKey as CNKeyDescriptor,
            CNContactJobTitleKey as CNKeyDescriptor,
        ]

        var allContacts: [CNContact] = []

        // Search by name
        if let predicate = try? CNContact.predicateForContacts(matchingName: query) as NSPredicate {
            if let contacts = try? contactStore.unifiedContacts(matching: predicate, keysToFetch: keys) {
                allContacts.append(contentsOf: contacts)
            }
        }

        // Search by email if it looks like one
        if query.contains("@") {
            let predicate = CNContact.predicateForContacts(matchingEmailAddress: query)
            if let contacts = try? contactStore.unifiedContacts(matching: predicate, keysToFetch: keys) {
                allContacts.append(contentsOf: contacts)
            }
        }

        // Search by phone if it looks like one
        let digits = query.filter { $0.isNumber }
        if digits.count >= 7 {
            let phoneNumber = CNPhoneNumber(stringValue: query)
            let predicate = CNContact.predicateForContacts(matching: phoneNumber)
            if let contacts = try? contactStore.unifiedContacts(matching: predicate, keysToFetch: keys) {
                allContacts.append(contentsOf: contacts)
            }
        }

        // Deduplicate
        var seen = Set<String>()
        var results: [[String: Any]] = []
        for contact in allContacts {
            if seen.contains(contact.identifier) { continue }
            seen.insert(contact.identifier)

            var entry: [String: Any] = [
                "given_name": contact.givenName,
                "family_name": contact.familyName,
            ]
            if !contact.organizationName.isEmpty { entry["organization"] = contact.organizationName }
            if !contact.jobTitle.isEmpty { entry["job_title"] = contact.jobTitle }

            let emails = contact.emailAddresses.map { $0.value as String }
            if !emails.isEmpty { entry["emails"] = emails }

            let phones = contact.phoneNumbers.map { $0.value.stringValue }
            if !phones.isEmpty { entry["phones"] = phones }

            if let bday = contact.birthday {
                entry["birthday"] = "\(bday.month ?? 0)-\(bday.day ?? 0)"
            }

            results.append(entry)
        }

        if results.isEmpty {
            return "{\"contacts\": [], \"note\": \"No contacts found for query.\"}"
        }

        return toJSON(results)
    }

    // MARK: - iMessage Handlers

    private func handleIMessageRecent(_ raw: String) -> String {
        let limit = min(parseIntParam(raw, name: "limit", fallback: 10), 50)

        let sql = """
            SELECT
                c.display_name,
                c.chat_identifier,
                m.text,
                datetime(m.date/1000000000 + 978307200, 'unixepoch', 'localtime') as msg_date,
                m.is_from_me
            FROM message m
            JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
            JOIN chat c ON c.ROWID = cmj.chat_id
            WHERE m.text IS NOT NULL AND m.text != ''
            ORDER BY m.date DESC
            LIMIT \(limit)
        """

        return querySQLiteDB(sql)
    }

    private func handleIMessageSearch(_ raw: String) -> String {
        let query = parseStringParam(raw, name: "q") ?? ""
        let limit = min(parseIntParam(raw, name: "limit", fallback: 20), 100)

        guard !query.isEmpty else {
            return "{\"error\": \"Missing required parameter: q\"}"
        }

        let sql = """
            SELECT
                c.display_name,
                c.chat_identifier,
                m.text,
                datetime(m.date/1000000000 + 978307200, 'unixepoch', 'localtime') as msg_date,
                m.is_from_me
            FROM message m
            JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
            JOIN chat c ON c.ROWID = cmj.chat_id
            WHERE m.text LIKE '%\(query.replacingOccurrences(of: "'", with: "''"))%'
            ORDER BY m.date DESC
            LIMIT \(limit)
        """

        return querySQLiteDB(sql)
    }

    private func handleIMessageConversation(_ raw: String) -> String {
        let contact = parseStringParam(raw, name: "contact") ?? ""
        let limit = min(parseIntParam(raw, name: "limit", fallback: 30), 100)

        guard !contact.isEmpty else {
            return "{\"error\": \"Missing required parameter: contact\"}"
        }

        let escaped = contact.replacingOccurrences(of: "'", with: "''")

        let sql = """
            SELECT
                c.display_name,
                c.chat_identifier,
                m.text,
                datetime(m.date/1000000000 + 978307200, 'unixepoch', 'localtime') as msg_date,
                m.is_from_me
            FROM message m
            JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
            JOIN chat c ON c.ROWID = cmj.chat_id
            WHERE m.text IS NOT NULL AND m.text != ''
                AND (c.display_name LIKE '%\(escaped)%'
                     OR c.chat_identifier LIKE '%\(escaped)%')
            ORDER BY m.date DESC
            LIMIT \(limit)
        """

        return querySQLiteDB(sql)
    }

    private func querySQLiteDB(_ sql: String, dbPath: String? = nil, errorHint: String = "Grant Full Disk Access to ZExpOverlay.") -> String {
        let path = dbPath ?? chatDBPath
        var db: OpaquePointer?
        guard sqlite3_open_v2(path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else {
            let err = db.flatMap { String(cString: sqlite3_errmsg($0)) } ?? "unknown"
            sqlite3_close(db)
            return "{\"error\": \"Cannot open database. \(errorHint) (\(err))\"}"
        }
        defer { sqlite3_close(db) }

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            let err = String(cString: sqlite3_errmsg(db!))
            return "{\"error\": \"SQL error: \(err)\"}"
        }
        defer { sqlite3_finalize(stmt) }

        var results: [[String: Any]] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            let colCount = sqlite3_column_count(stmt)
            var row: [String: Any] = [:]
            for i in 0..<colCount {
                let name = String(cString: sqlite3_column_name(stmt, i))
                switch sqlite3_column_type(stmt, i) {
                case SQLITE_TEXT:
                    row[name] = String(cString: sqlite3_column_text(stmt, i))
                case SQLITE_INTEGER:
                    row[name] = sqlite3_column_int64(stmt, i)
                case SQLITE_FLOAT:
                    row[name] = sqlite3_column_double(stmt, i)
                default:
                    row[name] = NSNull()
                }
            }
            results.append(row)
        }

        if results.isEmpty {
            return "{\"messages\": [], \"note\": \"No messages found.\"}"
        }

        return toJSON(results)
    }

    // MARK: - Mail Handlers

    private func handleMailRecent(_ raw: String) -> String {
        let limit = min(parseIntParam(raw, name: "limit", fallback: 10), 50)

        let sql = """
            SELECT
                s.subject,
                a.address AS sender,
                a.comment AS sender_name,
                datetime(m.date_received, 'unixepoch', 'localtime') AS received,
                m.read AS is_read,
                substr(m.summary, 1, 300) AS preview
            FROM messages m
            LEFT JOIN subjects s ON m.subject = s.ROWID
            LEFT JOIN addresses a ON m.sender = a.ROWID
            WHERE m.date_received IS NOT NULL
            ORDER BY m.date_received DESC
            LIMIT \(limit)
        """

        return querySQLiteDB(sql, dbPath: mailDBPath, errorHint: "Grant Full Disk Access to ZExpOverlay for Mail access.")
    }

    private func handleMailSearch(_ raw: String) -> String {
        let query = parseStringParam(raw, name: "q") ?? ""
        let limit = min(parseIntParam(raw, name: "limit", fallback: 20), 100)

        guard !query.isEmpty else {
            return "{\"error\": \"Missing required parameter: q\"}"
        }

        let escaped = query.replacingOccurrences(of: "'", with: "''")

        let sql = """
            SELECT
                s.subject,
                a.address AS sender,
                a.comment AS sender_name,
                datetime(m.date_received, 'unixepoch', 'localtime') AS received,
                m.read AS is_read,
                substr(m.summary, 1, 300) AS preview
            FROM messages m
            LEFT JOIN subjects s ON m.subject = s.ROWID
            LEFT JOIN addresses a ON m.sender = a.ROWID
            WHERE s.subject LIKE '%\(escaped)%'
               OR m.summary LIKE '%\(escaped)%'
               OR a.address LIKE '%\(escaped)%'
               OR a.comment LIKE '%\(escaped)%'
            ORDER BY m.date_received DESC
            LIMIT \(limit)
        """

        return querySQLiteDB(sql, dbPath: mailDBPath, errorHint: "Grant Full Disk Access to ZExpOverlay for Mail access.")
    }

    // MARK: - Helpers

    private func parseIntParam(_ raw: String, name: String, fallback: Int) -> Int {
        guard let range = raw.range(of: "\(name)=") else { return fallback }
        let start = range.upperBound
        let rest = raw[start...]
        if let end = rest.firstIndex(where: { !$0.isNumber }) {
            return Int(rest[start..<end]) ?? fallback
        }
        return Int(rest) ?? fallback
    }

    private func parseStringParam(_ raw: String, name: String) -> String? {
        guard let range = raw.range(of: "\(name)=") else { return nil }
        let start = range.upperBound
        let rest = raw[start...]
        let end = rest.firstIndex(where: { $0 == "&" || $0 == " " || $0 == "\r" || $0 == "\n" }) ?? rest.endIndex
        let value = String(rest[start..<end])
        return value.removingPercentEncoding ?? value
    }

    private func toJSON(_ obj: Any) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: obj),
              let str = String(data: data, encoding: .utf8) else {
            return "{\"error\": \"JSON serialization failed\"}"
        }
        return str
    }

    private func buildHTTPResponse(body: String) -> String {
        let contentLength = body.utf8.count
        return "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: \(contentLength)\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n\(body)"
    }
}
