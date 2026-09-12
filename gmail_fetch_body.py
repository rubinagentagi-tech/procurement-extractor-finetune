"""Fetch and print the body text of specific Gmail messages matching a query (read-only)."""
import email, imaplib, subprocess, sys
from email.header import decode_header

USER = "rubinagentagi@gmail.com"
PWD = subprocess.run(["secret-tool", "lookup", "service", "himalaya-gmail", "account", USER],
                     capture_output=True, text=True).stdout.strip()
QUERY = sys.argv[1]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def dec(s):
    if not s:
        return ""
    out = []
    for txt, enc in decode_header(s):
        out.append(txt.decode(enc or "utf-8", "replace") if isinstance(txt, bytes) else txt)
    return "".join(out)


def body_text(msg):
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    return part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", "replace")
                except Exception:
                    continue
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                try:
                    import re
                    html = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", "replace")
                    return re.sub(r"<[^>]+>", " ", html)
                except Exception:
                    continue
    else:
        try:
            return msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf-8", "replace")
        except Exception:
            return str(msg.get_payload())
    return ""


M = imaplib.IMAP4_SSL("imap.gmail.com")
M.login(USER, PWD)
M.select('"[Gmail]/All Mail"', readonly=True)
typ, data = M.search(None, QUERY)
ids = data[0].split()
print(f"matches: {len(ids)}")
for i in ids[-LIMIT:]:
    typ, d = M.fetch(i, "(RFC822)")
    msg = email.message_from_bytes(d[0][1])
    print("=" * 70)
    print("DATE:", dec(msg.get("Date")), "| FROM:", dec(msg.get("From")), "| SUBJ:", dec(msg.get("Subject")))
    txt = " ".join(body_text(msg).split())
    print(txt[:2500])
M.logout()
