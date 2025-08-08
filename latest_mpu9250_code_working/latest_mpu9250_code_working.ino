#include <Wire.h>
#include <MPU9250_asukiaaa.h>

MPU9250_asukiaaa mySensor;

void setup() {
  Serial.begin(115200);
  Wire.begin(22, 21); // ESP32 SDA, SCL

  mySensor.setWire(&Wire);
  mySensor.beginAccel();  // Initialize accelerometer

  Serial.println("timestamp_ms,ax,ay,az");  // CSV header
}

void loop() {
  mySensor.accelUpdate();

  unsigned long timestamp = millis();  // Milliseconds since boot

  // Print CSV: timestamp_ms, ax, ay, az
  Serial.print(timestamp); Serial.print(",");
  Serial.print(mySensor.accelX()); Serial.print(",");
  Serial.print(mySensor.accelY()); Serial.print(",");
  Serial.println(mySensor.accelZ());

  delay(20);  // 10 Hz
}
