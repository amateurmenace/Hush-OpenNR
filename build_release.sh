#!/bin/bash
# OpenNR — build a macOS release: universal .ofx bundle, double-clickable .pkg
# installer, and a zip fallback with a .command install script.
set -euo pipefail
cd "$(dirname "$0")"

VERSION="3.7.0"

echo "== building plugin =="
make -C plugin clean >/dev/null
make -C plugin

echo "== staging =="
rm -rf release
STAGE="release/stage"
mkdir -p "$STAGE"
cp -R plugin/OpenNR.ofx.bundle "$STAGE/"

echo "== signing =="
# Prefer a Developer ID Application identity when one is present; fall back to
# ad-hoc. (Notarization additionally requires --timestamp and hardened runtime.)
DEV_ID=$(security find-identity -v -p codesigning 2>/dev/null | grep -o '"Developer ID Application: [^"]*"' | head -1 | tr -d '"') || true
if [ -n "${DEV_ID:-}" ]; then
    echo "signing with: $DEV_ID"
    codesign --force --deep --timestamp --options runtime --sign "$DEV_ID" "$STAGE/OpenNR.ofx.bundle"
else
    echo "no Developer ID found — ad-hoc signing"
    codesign --force --deep --sign - "$STAGE/OpenNR.ofx.bundle"
fi
codesign --verify --deep "$STAGE/OpenNR.ofx.bundle"

echo "== sanity: minimum macOS of the shipped binary =="
otool -arch arm64 -l "$STAGE/OpenNR.ofx.bundle/Contents/MacOS/OpenNR.ofx" | grep -m1 minos

echo "== pkg installer =="
# Ticket the BUNDLE first. Everything we ship contains it, so both artifacts
# then inherit a stapled payload. Order is the whole trick: the old script
# notarized at the end and built the zip afterwards from an unstapled stage, so
# the zip shipped a ticket-less bundle even on a good run — which is why the
# README used to tell people to run `xattr -dr`.
NOTARY_PROFILE="${NOTARY_PROFILE:-opennr-notary}"
echo "== notarize =="
if ! xcrun notarytool history --keychain-profile "$NOTARY_PROFILE" >/dev/null 2>&1; then
    echo "FATAL: no notarytool profile '$NOTARY_PROFILE'."
    echo "  Store one, then re-run:"
    echo "    xcrun notarytool store-credentials $NOTARY_PROFILE --apple-id <id> --team-id 6M536MV7GT"
    echo "  (Refusing to ship un-notarized: that is how 3.6.0 and earlier went out"
    echo "   Gatekeeper-rejected while this script reported success.)"
    exit 1
fi
ditto -c -k --keepParent "$STAGE/OpenNR.ofx.bundle" "release/_submit.zip"
xcrun notarytool submit "release/_submit.zip" --keychain-profile "$NOTARY_PROFILE" --wait --timeout 30m
rm -f "release/_submit.zip"
xcrun stapler staple "$STAGE/OpenNR.ofx.bundle"
xcrun stapler validate "$STAGE/OpenNR.ofx.bundle"

echo "== pkg (built from the stapled bundle) =="
PKG_STAGE="release/pkgroot"
rm -rf "$PKG_STAGE"; mkdir -p "$PKG_STAGE"
cp -R "$STAGE/OpenNR.ofx.bundle" "$PKG_STAGE/"
pkgbuild --root "$PKG_STAGE" \
         --identifier org.opennr.plugin \
         --version "$VERSION" \
         --install-location "/Library/OFX/Plugins" \
         "release/OpenNR-$VERSION-macOS-unsigned.pkg"

# An unsigned pkg cannot be notarized — Apple rejects it at submission and the
# error reads like the notary profile is broken. So this is fatal, not a shrug.
INST_ID=$(security find-identity -v 2>/dev/null | grep -o '"Developer ID Installer: [^"]*"' | head -1 | tr -d '"') || true
if [ -z "${INST_ID:-}" ]; then
    echo "FATAL: no Developer ID Installer identity — the pkg would ship unsigned"
    echo "  and Gatekeeper would reject it on double-click. Create one at"
    echo "  developer.apple.com (Certificates → + → Developer ID Installer)."
    exit 1
fi
echo "signing pkg with: $INST_ID"
productsign --sign "$INST_ID" "release/OpenNR-$VERSION-macOS-unsigned.pkg" "release/OpenNR-$VERSION-macOS.pkg"
rm "release/OpenNR-$VERSION-macOS-unsigned.pkg"
# the pkg is its own artifact and needs its own ticket, payload notwithstanding
xcrun notarytool submit "release/OpenNR-$VERSION-macOS.pkg" --keychain-profile "$NOTARY_PROFILE" --wait --timeout 30m
xcrun stapler staple "release/OpenNR-$VERSION-macOS.pkg"

echo "== zip fallback (also from the stapled bundle) =="
cp "installer/Install OpenNR (macOS).command" "$STAGE/"
cp README.md LICENSE "$STAGE/"
rm -f "release/OpenNR-$VERSION-macOS.zip"
(cd "$STAGE" && zip -qry "../OpenNR-$VERSION-macOS.zip" .)

echo
echo "Gatekeeper verdict (what the user's Mac will say):"
spctl -a -t install -v "release/OpenNR-$VERSION-macOS.pkg" 2>&1 | sed 's/^/  pkg: /'
xcrun stapler validate "$STAGE/OpenNR.ofx.bundle" 2>&1 | tail -1 | sed 's/^/  zip payload: /'

echo
echo "Release artifacts:"
ls -la release/*.pkg release/*.zip
