// Ask for CSV file
var csvFile = File.openDialog("Select CSV file.", "CSV files:*.csv");
if (csvFile != null) {
  csvFile.open("r");
  
  // Read CSV data
  var csvData = csvFile.read().split("\n");
  var headers = csvData[0].split(",");
  var keyframes = [];

  var compWidth = app.project.activeItem.width;
  var compHeight = app.project.activeItem.height;

  var compSize = compWidth * compHeight

  
  for (var i = 1; i < csvData.length; i++) {
    var row = csvData[i].split(",");
    var frameTime = parseFloat(row[0]);
    var position = [parseFloat(row[1]), parseFloat(row[2]), 0];
    var rotation = [parseFloat(row[3]), parseFloat(row[4]), parseFloat(row[5])];
    var size = (parseFloat(row[6]) / compSize) * 100;
    
    // Create keyframe data
    var keyframe = {
      time: frameTime,
      position: position,
      orientation: rotation,
      scale: [size, size, size]
    };
    
    keyframes.push(keyframe);
  }
  
  csvFile.close();
  
  // Create 3D null object
  var nullLayer = app.project.activeItem.layers.addNull();
  nullLayer.threeDLayer = true;
  nullLayer.name = "3D Null";
  
  // Add position, rotation, and scale keyframes
  for (var i = 0; i < keyframes.length; i++) {
    try {
      var obj = keyframes[i];
      var time = obj.time;

      nullLayer.position.setValueAtTime(time, obj.position);

      nullLayer.xRotation.setValueAtTime(time, obj.orientation[0]);
      nullLayer.yRotation.setValueAtTime(time, obj.orientation[1]);
      nullLayer.zRotation.setValueAtTime(time, obj.orientation[2]);
      
      nullLayer.scale.setValueAtTime(time, obj.scale);
    }
    catch(x_x){
      continue;
    }
    
  }
}
