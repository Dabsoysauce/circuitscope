

# General libraries
import time
# Libraries for the PWM driver with servos (uses BCM numbering)
from adafruit_servokit import ServoKit


# Specify the channels you are using on the PWM driver
channel_lift_left   = 0
channel_lift_right  = 1
channel_sweep_left  = 2
channel_sweep_right = 3


# Initialize ServoKit for the PWM board (fixes the pwm frequency to 50)
kit = ServoKit(channels=16)

# Set the servo range to 180 degrees for all four servos
kit.servo[channel_lift_left].set_pulse_width_range(400, 2300)
kit.servo[channel_lift_right].set_pulse_width_range(400, 2300)
kit.servo[channel_sweep_left].set_pulse_width_range(400, 2300)
kit.servo[channel_sweep_right].set_pulse_width_range(400, 2300)


# Gait angles. If a leg moves the wrong way, swap that side's two numbers.
# Lift servos (0,1): up = leg raised, down = leg planted
LIFT_LEFT_UP    = 120
LIFT_LEFT_DOWN  = 60
LIFT_RIGHT_UP   = 120
LIFT_RIGHT_DOWN = 60
# Sweep servos (2,3): fwd = leg reached forward, back = leg pushed behind
SWEEP_LEFT_FWD   = 120
SWEEP_LEFT_BACK  = 60
SWEEP_RIGHT_FWD  = 120
SWEEP_RIGHT_BACK = 60


# Helper functions for the two leg motions
def legs_up():
    kit.servo[channel_lift_left].angle  = LIFT_LEFT_UP
    kit.servo[channel_lift_right].angle = LIFT_RIGHT_UP

def legs_down():
    kit.servo[channel_lift_left].angle  = LIFT_LEFT_DOWN
    kit.servo[channel_lift_right].angle = LIFT_RIGHT_DOWN

def sweep_forward():
    kit.servo[channel_sweep_left].angle  = SWEEP_LEFT_FWD
    kit.servo[channel_sweep_right].angle = SWEEP_RIGHT_FWD

def sweep_back():
    kit.servo[channel_sweep_left].angle  = SWEEP_LEFT_BACK
    kit.servo[channel_sweep_right].angle = SWEEP_RIGHT_BACK


# Start pose: legs down and swept back (end of a power stroke)
legs_down()
sweep_back()


# Initialize the finite state machine
FSM1State = 0
FSM1NextState = 0
FSM1LastTime = 0

# Time to hold each phase (seconds). Lower = faster walk, but give the
# servos enough time to actually reach each pose.
duration = 0.4


try:
    print("Press CTRL+C to end the program.")

    while True:

        # Check the current time
        currentTime = time.time()

        # Update the state
        FSM1State = FSM1NextState


        # State 0: lift the legs up (clear the ground)
        if (FSM1State == 0):
            if (currentTime - FSM1LastTime > duration):
                legs_up()
                print("Lift legs up")
                FSM1NextState = 1
            else:
                FSM1NextState = 0

        # State 1: sweep the legs forward while raised (recovery)
        elif (FSM1State == 1):
            if (currentTime - FSM1LastTime > duration):
                sweep_forward()
                print("Sweep legs forward")
                FSM1NextState = 2
            else:
                FSM1NextState = 1

        # State 2: plant the legs down (grip the floor)
        elif (FSM1State == 2):
            if (currentTime - FSM1LastTime > duration):
                legs_down()
                print("Plant legs down")
                FSM1NextState = 3
            else:
                FSM1NextState = 2

        # State 3: sweep the legs back on the ground (power stroke)
        elif (FSM1State == 3):
            if (currentTime - FSM1LastTime > duration):
                sweep_back()
                print("Power stroke -> move forward")
                FSM1NextState = 0
            else:
                FSM1NextState = 3

        # State ??
        else:
            print("Error: unrecognized state for FSM1")
            break


        # If there is a state change, record the time
        if (FSM1State != FSM1NextState):
            FSM1LastTime = currentTime


# Quit the program when the user presses CTRL + C
except KeyboardInterrupt:
    pass
finally:
    # Relax all four servos
    kit.servo[channel_lift_left].angle   = None
    kit.servo[channel_lift_right].angle  = None
    kit.servo[channel_sweep_left].angle  = None
    kit.servo[channel_sweep_right].angle = None

