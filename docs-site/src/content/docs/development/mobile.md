# Mobile (Android)

The AgentX client is built with Tauri v2, which targets Android in addition to desktop
(v0.20.0 is the "Mobile-Ready Alpha"). The Android app is the same React UI wrapped in a Tauri
WebView; it talks to an AgentX API server over HTTP, exactly like the desktop client.

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Android SDK & NDK | Install via Android Studio or the command-line tools |
| JDK 17+ | Required by the Gradle build |
| Rust (stable) | Tauri native compilation |
| Environment | `ANDROID_HOME`, `NDK_HOME`, `JAVA_HOME` must point at your installs |

See the [Tauri Android prerequisites](https://v2.tauri.app/start/prerequisites/#android) for
the authoritative setup.

## Tasks

```bash
task client:android:init     # one-time: bootstrap the Tauri Android project
task client:dev:android      # run on a connected device/emulator with hot reload
task client:build:android    # build a debug-signed APK
```

These wrap `bunx tauri android init | dev | build --apk true --debug` (run from `client/`).
`client:android:init` generates the Gradle project under
`client/src-tauri/gen/android/` — run it once before the other two.

The Android target is configured in `client/src-tauri/tauri.conf.json` (`android.minSdkVersion`
is 24; app identifier `com.redacted.agentx-client`, version synced from `versions.yaml`).

## Installing the APK

`client:build:android` outputs to:

```
client/src-tauri/gen/android/app/build/outputs/apk/universal/debug/app-debug.apk
```

Install it on a connected device:

```bash
adb install client/src-tauri/gen/android/app/build/outputs/apk/universal/debug/app-debug.apk
```

!!! note "Debug signing"
    The APK is signed with Gradle's auto-generated debug keystore — fine for testing, but not
    suitable for Play Store distribution without re-signing with a release key.

## Connecting to a server

The mobile app has **no hard-coded API URL**. On launch it reads saved servers from local
storage (`getActiveServer()` in `client/src/lib/api/core.ts`); if none exist it falls back to
the build-time `VITE_API_URL`, otherwise `http://localhost:12319`. In practice you add your
server's URL through the in-app connection screen.

Because `localhost` on the phone is the phone itself, point the app at the API server's
LAN/public address — typically a [cluster gateway](../deployment/clusters.md) URL. Requests
carry the `X-Auth-Token` header when [authentication](../deployment/authentication.md) is on,
plus the optional `AgentX-Gateway-Token` when connecting through a gateway.
