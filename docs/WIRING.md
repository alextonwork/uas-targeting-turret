# Wiring and power

The single most common failure on Pi gimbal builds is powering servos from the Pi. Two
MG996R-class servos can each pull 1-2.5 A at stall, and the inrush when they start
moving sags the 5 V rail enough to brown out the Pi, corrupt the SD card, or reset the
camera mid-frame. Give the servos **their own supply** and share only ground.

## Block diagram

```
                 ┌──────────────────────────┐
  5.1 V 5 A USB-C│ Raspberry Pi 4 / 5       │
  (Pi PSU) ─────▶│                          │ CSI ribbon
                 │  GPIO2 SDA ──────────┐   │◀──────────── Camera module
                 │  GPIO3 SCL ────────┐ │   │
                 │  3V3 ────────────┐ │ │   │
                 │  GND ──────────┐ │ │ │   │
                 └────────────────┼─┼─┼─┼───┘
                                  │ │ │ │
                 ┌────────────────┼─┼─┼─┼──────────────────┐
                 │ PCA9685        GND VCC SCL SDA           │
                 │  (logic side powered from Pi 3V3)        │
                 │                                          │
                 │  V+ terminal ◀── servo rail +6.0 V       │
                 │  GND terminal ◀─ servo rail GND ─────────┼──┐
                 │                                          │  │
                 │  CH0 ── pan servo  (sig / V+ / GND)      │  │
                 │  CH1 ── tilt servo (sig / V+ / GND)      │  │
                 └──────────────────────────────────────────┘  │
                                                               │
  Servo rail: 6 V 5 A BEC / bench supply ─── +  ── PCA9685 V+  │
                                         └── −  ── PCA9685 GND ┘── (joined to Pi GND
                                                                   through the PCA9685
                                                                   GND pins: ONE common
                                                                   ground, no loops)
  Bulk cap: 1000-2200 uF / 10 V+ electrolytic across V+ and GND at the PCA9685 terminal.
```

## Parts and ratings

| Item | Recommendation | Why |
|---|---|---|
| Servo supply | 6.0 V, ≥ 5 A UBEC or bench PSU | 2 × MG996R stall ≈ 5 A; 6 V gives faster slew than 5 V |
| Bulk capacitor | 1000–2200 µF, ≥ 10 V, at the PCA9685 V+ terminal | Absorbs start-up current spikes so the rail doesn't dip |
| Servo wire | 20–22 AWG for runs over 30 cm | Voltage drop at 2 A in thin leads causes jitter |
| Pi supply | Official 5.1 V 3 A (Pi 4) / 5 A (Pi 5) | Separate from the servo rail |
| Fuse (optional) | 5 A blade fuse on servo + | Protects against a stalled or shorted servo |

## Rules

1. **Never** connect the servo rail + to the Pi 5 V pins, and never power servos from the
   Pi's 5 V header.
2. **Do** connect grounds: servo rail GND ↔ PCA9685 GND ↔ Pi GND. Without a common
   ground the PWM signal has no reference and servos twitch randomly.
3. Power the PCA9685 *logic* (VCC pin) from the Pi's **3.3 V** so I²C levels match the Pi.
4. Power-up order: Pi first, start the software (it centres the servos), then switch on
   the servo rail. Power-down: servo rail off first. This avoids the servos slamming to
   an end stop while the PWM output is undefined.
5. Mechanically limit travel, or at minimum set `control.*.min_angle/max_angle` so the
   camera cable can't be wound up or yanked out.

## pigpio wiring (no PCA9685)

If you use `servo.driver: pigpio`, connect servo signal wires directly to GPIO18 (pan)
and GPIO13 (tilt) (hardware-PWM capable pins), power servos from the dedicated rail
exactly as above, and keep the common ground. Run `sudo systemctl enable --now pigpiod`.
The 3.3 V GPIO signal drives almost every hobby servo fine.

## Checking the rail

With both servos sweeping (`python -m turret --mode live --set servo.driver=pca9685` and
a target moving in front of the camera, or the servo test in the README), measure the
servo rail at the PCA9685 terminal. It should stay above ~5.6 V. Run `vcgencmd
get_throttled` on the Pi afterwards; anything other than `throttled=0x0` means the Pi
saw under-voltage and its supply needs attention.
