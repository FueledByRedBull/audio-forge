# Security policy

Security fixes target the latest stable release. A release candidate remains
prerelease software until its published qualification evidence is complete;
older versions receive no promised backports.

Report a suspected vulnerability using GitHub's **Security > Report a
vulnerability** for this repository when available. If private reporting is
unavailable, open a public issue requesting a private contact without including
the exploit, sensitive data, or credentials. This project does not promise a
response-time SLA.

Include affected versions, a minimal reproduction, impact, and whether the
problem requires an explicitly enabled external model/DLL override. Share
privacy-safe diagnostics only; never include audio recordings, authentication
tokens, environment files, or private device names by default.

Release checks include RustSec, locked Python dependency audits, reviewed
Semgrep findings, and exact-artifact provenance. These checks do not certify
the absence of vulnerabilities. Native code and opted-in external DLLs run
with the application's permissions; external overrides must come from trusted
sources.
