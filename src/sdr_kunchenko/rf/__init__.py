"""RF I/O and OcuSync packet processing for the POC pelengator (Phase 7).

Stream A modules (Tampere offline proxy):
    packet_detector       — streaming GSA-CUSUM burst detection
    ocusync_parser        — OcuSync-1 header inspection / ID extraction (optional)
    dual_channel_simulator — synthetic dual-antenna geometry on single-channel IQ
    fingerprint           — DSGE+spectral SEI filter (Plan B path)

Stream B modules (live SDR, added when hardware arrives):
    live_source           — uhd.usrp.MultiUSRP wrapper
    calibration           — IQ-imbalance + phase sync on a tee signal
"""
