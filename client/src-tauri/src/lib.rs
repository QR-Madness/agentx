// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

/// Enable + auto-grant the webview's microphone (getUserMedia) on Linux.
///
/// webkit2gtk ships with `enable-media-stream` OFF (so `navigator.mediaDevices`
/// doesn't even work) and denies the `permission-request` for user media by
/// default — both must be handled natively, otherwise the Ambassador's
/// push-to-talk voice input silently does nothing on the packaged Linux app.
/// macOS/Windows webviews handle media-capture permission themselves (macOS via
/// the bundled Info.plist usage string + audio-input entitlement).
#[cfg(target_os = "linux")]
fn enable_webview_microphone(app: &tauri::App) {
    use tauri::Manager;
    use webkit2gtk::glib::prelude::*;
    use webkit2gtk::{PermissionRequestExt, SettingsExt, WebViewExt};

    let Some(window) = app.get_webview_window("main") else {
        return;
    };
    let _ = window.with_webview(|webview| {
        let wv = webview.inner();
        if let Some(settings) = WebViewExt::settings(&wv) {
            settings.set_enable_media_stream(true);
            settings.set_enable_mediasource(true);
        }
        // Grant only user-media (mic/camera) requests from our own content; let
        // anything else fall through to the default (deny) handling.
        wv.connect_permission_request(|_wv, request| {
            if request
                .downcast_ref::<webkit2gtk::UserMediaPermissionRequest>()
                .is_some()
            {
                request.allow();
                true
            } else {
                false
            }
        });
    });
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            #[cfg(target_os = "linux")]
            enable_webview_microphone(app);
            #[cfg(not(target_os = "linux"))]
            let _ = app;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
