from machine import Pin
import time

led = Pin("LED", Pin.OUT)

while True:
    led.toggle()
    print("LED toggle")
    time.sleep(0.5)