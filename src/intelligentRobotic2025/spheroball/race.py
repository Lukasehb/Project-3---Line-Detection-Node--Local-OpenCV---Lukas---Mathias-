# ...existing code...
import sys
import time
import math
import argparse
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color
from spherov2.commands.power import Power

# --- Configuration (tweak after real-world calibration) ---
# distances in cm and headings in degrees (clockwise lap)
SEGMENTS_CM = [200, 200, 100, 100, 150, 100, 100, 200, 250]
HEADINGS_DEG = [0, 90, 180, 270, 180, 90, 180, 270, 0]

# default speed percent (0-100) and estimated cm/s for conversion
DEFAULT_SPEED_PCT = 70
DEFAULT_CM_PER_SEC = 41.7

# motion profile (seconds)
RAMP_SEC = 0.35
BRAKE_SEC = 0.20

# LED colours
LED_READY = Color(0, 0, 200)
LED_RUNNING = Color(255, 120, 0)
LED_OK = Color(0, 200, 0)
LED_ERROR = Color(200, 0, 0)

# safety
DISCOVER_TIMEOUT = 15.0


def find_toy(name_or_mac: str):
    toys = scanner.find_toys()
    for t in toys:
        if getattr(t, "name", "") == name_or_mac or getattr(t, "address", "") == name_or_mac:
            return t
    return None


def secs_for_cm(dist_cm: float, cm_per_s: float) -> float:
    cmps = max(0.1, cm_per_s)
    return dist_cm / cmps


def read_battery(toy):
    try:
        v = Power.get_battery_voltage(toy)
        return v
    except Exception:
        return None


def gentle_move(api: SpheroEduAPI, heading: int, speed_pct: int, duration: float):
    """
    Smooth start/cruise/stop behaviour.
    Tries api.roll(heading, speed_pct, seconds) if available; otherwise falls back to set_heading/set_speed with sleeps.
    """
    if duration <= 0:
        return

    # try high-level roll API if present
    try:
        roll_fn = getattr(api, "roll", None)
        if callable(roll_fn):
            # ramp up
            api.roll(heading, speed_pct, RAMP_SEC)
            cruise = max(0.0, duration - 2 * RAMP_SEC)
            if cruise > 0:
                api.roll(heading, speed_pct, cruise)
            # ramp down
            api.roll(heading, max(10, int(0.4 * speed_pct)), RAMP_SEC)
            api.roll(heading, 0, BRAKE_SEC)
            return
    except Exception:
        pass

    # fallback: manual speed control
    try:
        api.set_heading(heading)
    except Exception:
        pass

    # ramp up
    try:
        api.set_speed(int(speed_pct / 100.0 * 255))
    except Exception:
        # if set_speed not available, ignore
        pass
    time.sleep(max(0.0, RAMP_SEC))

    # cruise
    cruise = max(0.0, duration - 2 * RAMP_SEC)
    if cruise > 0:
        time.sleep(cruise)

    # ramp down then stop
    try:
        api.set_speed(int(max(10, 0.4 * speed_pct) / 100.0 * 255))
    except Exception:
        pass
    time.sleep(max(0.0, RAMP_SEC))
    try:
        api.set_speed(0)
    except Exception:
        pass
    time.sleep(BRAKE_SEC)


def calibrate_zero(api: SpheroEduAPI):
    """
    Offer a simple calibration: call start/finish calibration if available, otherwise set heading to 0 after Enter.
    """
    print("Kalibratie: zet robot zo dat '0°' naar het eerste rechte stuk wijst.")
    try:
        start_cal = getattr(api, "start_calibration", None)
        if callable(start_cal):
            start_cal()
    except Exception:
        pass

    input("Druk Enter om te bevestigen (kalibratie wordt beëindigd)...")

    try:
        finish_cal = getattr(api, "finish_calibration", None)
        if callable(finish_cal):
            finish_cal()
        else:
            # fallback: force heading 0
            api.set_heading(0)
    except Exception:
        try:
            api.set_heading(0)
        except Exception:
            pass
    print("Kalibratie voltooid.")


def run_lap(api: SpheroEduAPI, segments_cm, headings_deg, cm_per_s: float, speed_pct: int):
    # prepare robot
    try:
        api.set_stabilization(True)
    except Exception:
        pass
    try:
        # set back/main leds if available
        if hasattr(api, "set_back_led"):
            api.set_back_led(255)
        if hasattr(api, "set_main_led"):
            api.set_main_led(LED_RUNNING)
        else:
            # fall back to front LED if present
            if hasattr(api, "set_front_led"):
                api.set_front_led(LED_RUNNING)
    except Exception:
        pass

    # countdown
    for k in (3, 2, 1):
        print(f"Start over {k}...")
        try:
            if hasattr(api, "set_main_led"):
                api.set_main_led(Color(255, 255, 0))
            time.sleep(0.35)
            if hasattr(api, "set_main_led"):
                api.set_main_led(Color(0, 0, 0))
            time.sleep(0.4)
        except Exception:
            time.sleep(0.75)
    print("GO!")
    try:
        if hasattr(api, "set_main_led"):
            api.set_main_led(LED_RUNNING)
    except Exception:
        pass

    t0 = time.perf_counter()
    for hdg, dist in zip(headings_deg, segments_cm):
        secs = secs_for_cm(dist, cm_per_s)
        print(f"→ {dist:.0f} cm  |  heading {int(hdg):3d}°  | est {secs:.2f}s")
        # set heading first so turn is executed while stopped (tight corners)
        try:
            api.set_heading(int(hdg))
        except Exception:
            pass

        # short pause to make turn visible/stable
        time.sleep(0.35)

        gentle_move(api, int(hdg), max(5, min(100, speed_pct)), secs)

        # ensure fully stopped before next corner
        try:
            api.set_speed(0)
        except Exception:
            pass
        time.sleep(0.12)

    # stop and show result
    try:
        if hasattr(api, "roll"):
            api.roll(0, 0, 0.05)
    except Exception:
        pass

    t1 = time.perf_counter()
    lap_time = t1 - t0

    # LED OK
    try:
        if hasattr(api, "set_back_led"):
            api.set_back_led(0)
        if hasattr(api, "set_main_led"):
            api.set_main_led(LED_OK)
        elif hasattr(api, "set_front_led"):
            api.set_front_led(LED_OK)
    except Exception:
        pass

    print(f"\nLap tijd: {lap_time:.3f} s")
    return lap_time


def discover_by_name(name: str, timeout: float = DISCOVER_TIMEOUT):
    print(f"Zoek SPHERO '{name}' (max {timeout:.0f}s)...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        toy = find_toy(name)
        if toy:
            print(f"Gevonden: {toy.name}")
            return toy
        time.sleep(0.5)
    return None


def main():
    parser = argparse.ArgumentParser(description="Autonome Sphero BOLT ronde (wijzerszin).")
    parser.add_argument("--name", "-n", required=True, help="Sphero naam of MAC (bv. SB-9DD8)")
    parser.add_argument("--speed", "-s", type=int, default=DEFAULT_SPEED_PCT, help="Snelheid in % (10-100)")
    parser.add_argument("--cmps", type=float, default=DEFAULT_CM_PER_SEC, help="Schaatting cm/s bij opgegeven speed")
    parser.add_argument("--segments", type=str, default=",".join(str(x) for x in SEGMENTS_CM),
                        help="Komma-gescheiden segmentlengtes (cm)")
    args = parser.parse_args()

    segments = [float(x) for x in args.segments.split(",") if x.strip()]
    headings = HEADINGS_DEG if len(segments) == len(HEADINGS_DEG) else HEADINGS_DEG[:len(segments)]

    toy = discover_by_name(args.name)
    if toy is None:
        print(f"Geen Sphero gevonden met naam {args.name}")
        sys.exit(1)

    try:
        with SpheroEduAPI(toy) as api:
            # ready
            try:
                if hasattr(api, "set_main_led"):
                    api.set_main_led(LED_READY)
                elif hasattr(api, "set_front_led"):
                    api.set_front_led(LED_READY)
            except Exception:
                pass

            # battery info
            bv = read_battery(toy)
            if bv is not None:
                print(f"Battery: {bv:.2f} V")
                if bv <= 3.5:
                    print("LET OP: lage batterijspanning")

            # calibration (user aligns; Enter to continue)
            calibrate_zero(api)

            input("Plaats robot achter start/finish-lijn en druk Enter om te starten (timer start bij vertrek)...")
            # run lap
            run_lap(api, segments, headings, args.cmps, args.speed)

            # finish LED
            try:
                if hasattr(api, "set_main_led"):
                    api.set_main_led(LED_OK)
            except Exception:
                pass

    except KeyboardInterrupt:
        print("\nAfgebroken door gebruiker.")
        try:
            with SpheroEduAPI(toy) as api:
                if hasattr(api, "roll"):
                    api.roll(0, 0, 0.2)
                if hasattr(api, "set_main_led"):
                    api.set_main_led(LED_ERROR)
        except Exception:
            pass
    except Exception as e:
        print(f"Fout: {e}")
        try:
            with SpheroEduAPI(toy) as api:
                if hasattr(api, "roll"):
                    api.roll(0, 0, 0.2)
                if hasattr(api, "set_main_led"):
                    api.set_main_led(LED_ERROR)
        except Exception:
            pass
        sys.exit(2)


if __name__ == "__main__":
    main()
# ...existing code...