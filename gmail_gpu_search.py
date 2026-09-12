"""Read-only Gmail search for GPU-cloud receipts/traces (IMAP, app password from keyring)."""
import email, imaplib, subprocess, sys
from email.header import decode_header

USER = "rubinagentagi@gmail.com"
PWD = subprocess.run(
    ["secret-tool", "lookup", "service", "himalaya-gmail", "account", USER],
    capture_output=True, text=True).stdout.strip()
if not PWD:
    sys.exit("no app password from keyring")

QUERY = sys.argv[1] if len(sys.argv) > 1 else "(X-GM-RAW \"runpod OR vast.ai OR lambdalabs OR paperspace OR tensordock OR coreweave OR 'gpu rental' OR 'gpu cloud'\")"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 40


def dec(s):
    if not s:
        return ""
    parts = []
    for txt, enc in decode_header(s):
        parts.append(txt.decode(enc or "utf-8", "replace") if isinstance(txt, bytes) else txt)
    return "".join(parts)


M = imaplib.IMAP4_SSL("imap.gmail.com")
M.login(USER, PWD)
M.select('"[Gmail]/All Mail"', readonly=True)
typ, data = M.search(None, QUERY)
ids = data[0].split()
print(f"matches: {len(ids)}")
rows = []
for i in ids[-LIMIT:]:
    typ, d = M.fetch(i, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])")
    raw = d[0][1].decode("utf-8", "replace")
    msg = email.message_from_string(raw)
    rows.append((dec(msg.get("Date")), dec(msg.get("From")), dec(msg.get("Subject"))))
rows.sort(key=lambda r: r[0])
for date, frm, subj in rows:
    print(f"{date[:31]:33} | {frm[:42]:44} | {subj[:70]}")
M.logout()
