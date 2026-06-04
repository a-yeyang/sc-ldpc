"""Download the 5G NR (3GPP TS 38.212) base-graph matrices into ./data.

Source: https://github.com/manuts/NR-LDPC-BG   (files NR_{bg}_{iLS}_{Z}.txt,
shift values already reduced mod Z).  Only a few (bg, iLS, Z) are needed for the
demo; edit FILES to fetch more.
"""
import os
import ssl
import urllib.request

BASE = "https://raw.githubusercontent.com/manuts/NR-LDPC-BG/master/"
FILES = ["NR_2_1_24.txt", "NR_2_0_16.txt", "NR_2_6_52.txt", "NR_1_1_24.txt", "README.txt"]


def main():
    os.makedirs("data", exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE     # some minimal Python installs lack CA certs
    for f in FILES:
        try:
            data = urllib.request.urlopen(BASE + f, timeout=30, context=ctx).read()
            open(os.path.join("data", f), "wb").write(data)
            print(f"OK   {f}  ({len(data)} bytes)")
        except Exception as e:
            print(f"FAIL {f}: {e!r}")


if __name__ == "__main__":
    main()
