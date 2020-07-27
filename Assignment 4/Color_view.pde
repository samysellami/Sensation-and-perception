import processing.serial.*; 

String buff = "";
int val = 0;
int wRed, wGreen, wBlue;

Serial port; 

void setup(){ 

  size(900,600);


  port = new Serial(this, "/dev/ttyUSB0", 9600); //remember to replace COM20 with the appropriate serial port on your computer
} 

void draw(){ 
  background(wRed,wGreen,wBlue);
  // check for serial, and process
  while (port.available() > 0) {
    serialEvent(port.read());
  }
} 

void serialEvent(int serial) { 

  if(serial != '\n') { 
    buff += char(serial);
  } 
  else {
    int cRed = buff.indexOf("R");
    int cGreen = buff.indexOf("G");
    int cBlue = buff.indexOf("B");

    if(cRed >=0){
      String val = buff.substring(cRed+3);
      wRed = Integer.parseInt(val.trim());
    }    
    if(cGreen >=0){
      String val = buff.substring(cGreen+3);
      wGreen = Integer.parseInt(val.trim());
    }
    if(cBlue >=0){
      String val = buff.substring(cBlue+3);
      wBlue = Integer.parseInt(val.trim());
    }

    buff = "";
  }
} 
