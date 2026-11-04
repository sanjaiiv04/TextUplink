# Ray-Ban glasses → Mac → Brave → Streamlit microphone setup

Verified with this project on November 3, 2026. The microphone worked after both
macOS and Brave were explicitly set to the glasses. The tested Bluetooth device
name was **RB Meta 0180**; another pair will have a different name.

## 1. Complete initial phone setup

- Charge the glasses and case.
- Pair the glasses with the **Meta AI app** on your phone and finish firmware updates.
- The Meta Horizon / Quest developer app used for VR is separate.
- Pairing with the phone alone does not make the glasses available to a Mac browser.

## 2. Pair with the Mac

- Temporarily turn off Bluetooth in the iPhone's **Settings** to prevent it from
  reconnecting during the Mac test. Keep the existing pairing; do not factory reset.
- Power on the glasses, take them out of the case, and fold both arms.
- Hold the **capture button on top of the frame** for about **12 seconds**, until
  the light **inside the glasses blinks blue**.
- Open **Mac System Settings → Bluetooth**. Select the glasses under nearby devices
  and click **Connect**. Confirm their status becomes connected.

The rear case button did not produce the expected blue pairing light during this
setup; it showed white. The capture-button method above successfully enabled pairing.
Follow your model's instructions if the controls differ. Avoid holding buttons
indefinitely or performing a factory reset as a troubleshooting shortcut.

Official pairing reference:
https://www.ray-ban.com/usa/c/frequently-asked-questions-ray-ban-meta-smart-glasses

## 3. Select and verify the Mac microphone

1. Open **System Settings → Sound → Input**.
2. Select the Ray-Ban glasses, such as **RB Meta 0180**.
3. Speak near the glasses. Confirm that the **input-level meter moves**.

If the glasses are missing, check the Mac Bluetooth connection first. If they are
listed but the meter does not move, troubleshoot the hardware connection before
recording in Streamlit.

## 4. Set playback separately

Open **System Settings → Sound → Output** and select either:

- **MacBook Speakers**, to listen through the computer; or
- The **Ray-Ban glasses**, to listen through their speakers.

Raise the selected output's volume and check mute. Input and output are separate
settings: keep input on the glasses when changing playback output.

## 5. Explicitly select the microphone in Brave

1. Type **brave://settings/content/microphone** into Brave's **address bar** and press Enter.
   This is a browser settings address, not a terminal command or web search.
2. In the **Microphone** dropdown, explicitly select the Ray-Ban device.
   On the tested setup it appeared as **RB Meta 0180 (Bluetooth) (System default)**.
3. Confirm that **Sites can ask to use your microphone** is enabled.
4. Allow microphone access for the Streamlit address you are using, normally
   **http://localhost:8501**. If the site is blocked, change its microphone permission
   through the site's settings. Another address or port may need separate permission.
5. Return to Streamlit and **reload the page** before making a fresh recording.

**The issue we encountered:** macOS input was already set correctly and its input
meter moved, but Brave still had **MacBook Air Microphone (Built-in)** selected.
Changing Brave to the glasses, reloading Streamlit, and making a new recording
resolved the test. System input selection alone was not sufficient.

## 6. Record, play and save in Streamlit

From the project folder:

```sh
source .venv/bin/activate
streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

1. Open the local URL in Brave, then choose the **Microphone test** tab.
2. Start recording and allow microphone access if prompted.
3. Speak for roughly **10 seconds**, with the Mac a few feet away and your voice
   near the glasses. Disconnect other headsets to avoid confusing input routes.
4. Press **Stop recording**.
5. Check **Recorded peak** and **Average recording level**, then press Play.
6. Click **Save microphone recording as WAV** to download `microphone_test.wav`.
7. Optionally upload that WAV in **Expressive captions** for isolation/transcription.

This test records the browser-selected microphone. It does not automatically verify
its hardware identity. The WAV is the original browser recording, with no isolation
or AI processing applied in this tab.

## If the recording is silent

Check in this order:

1. **Bluetooth:** does the Mac show the glasses connected?
2. **Mac input:** are the glasses selected, and does the level meter move when speaking?
3. **Brave input:** is the glasses device explicitly selected, rather than the built-in mic?
4. **Permissions:** can Brave use the microphone in macOS Privacy & Security → Microphone,
   and can the Streamlit site use it in Brave?
5. **Fresh capture:** reload Streamlit, then record again. An older silent recording
   will not change when input settings change.
6. **Recorded levels:** “digital silence” means every recorded sample is zero.
   A very quiet signal can also be inaudible; nonzero audio does not prove speech was captured.
7. **Playback:** check output device, mute and volume. Try MacBook Speakers to distinguish
   playback routing from capture. Hearing a system sound through the glasses confirms
   their speakers work, but does not establish that their microphone is being recorded.

When testing another pair, repeat the Mac and browser selections using its new
Bluetooth name. Recheck after reconnecting devices or switching browsers/profiles.
For phone-only use, this Mac setup is not an audio bridge from the iPhone; a separate
phone capture path would be needed.
