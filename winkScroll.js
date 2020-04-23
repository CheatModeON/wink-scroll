
let DEBUG = true

var leftEye
var resizedLeftEye

var text = document.createElement("P");
text.id = "txt";

var video = document.createElement("VIDEO");
video.id = "video";
video.width = 749
video.height = 560
video.autoplay = true;
video.defaultMuted = true;
video.style.position = "absolute";
video.style.top = 0 + "px";
video.style.left = 0 + "px";

if(!DEBUG){
  video.style.left = -window.innerWidth + "px";
}

video.load();
var canvas2 = document.createElement("CANVAS");
canvas2.id = "canvas2"
canvas2.width = 300
canvas2.height = 150
canvas2.style.position = "absolute";
canvas2.style.top = 570 + "px";
canvas2.style.left = 10 + "px";
var canvas3 = document.createElement("CANVAS");
canvas3.id = "canvas3"
canvas3.width = canvas2.width
canvas3.height = canvas2.height
canvas3.style.position = "absolute";
canvas3.style.top = 730 + "px";
canvas3.style.left = 10 + "px";

document.body.appendChild(video);
document.body.appendChild(canvas2);
document.body.appendChild(canvas3);
document.body.appendChild(text);

// https://docs.opencv.org/3.4/df/df7/tutorial_js_table_of_contents_setup.html

/*
const video = document.getElementById('video')
const canvas2 = document.getElementById('canvas2')
const canvas3 = document.getElementById('canvas3')
const text = document.getElementById('text')
*/

var bufferX = [0,0,0,0,0]
var bufferY = [0,0,0,0,0]
const bufferSize = 5;
var flag = 0;
var threshold = 35;

if(!DEBUG){
  canvas2.style.display="none";
  canvas3.style.display="none";
}

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('/models'), //change this to /models in order to work in your local folder
  faceapi.nets.faceLandmark68Net.loadFromUri('models'), //change this to /models in order to work in your local folder
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'), //change this to /models in order to work in your local folder
  faceapi.nets.faceExpressionNet.loadFromUri('/models') //change this to /models in order to work in your local folder
]).then(startVideo)

function startVideo() {
  navigator.getUserMedia(
    { video: {} },
    stream => video.srcObject = stream,
    err => console.error(err)
  )
}
let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
let dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);
let cap = new cv.VideoCapture(video);
const LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
const RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

video.addEventListener('play', () => {
  const canvas = faceapi.createCanvasFromMedia(video)

  canvas.style.position = "absolute"
  canvas.style.top = 0 + "px"
  canvas.style.left = 0 + "px"

  document.body.append(canvas)
  const displaySize = { width: video.width, height: video.height }
  faceapi.matchDimensions(canvas, displaySize)

  var ctx = canvas2.getContext("2d");
  ctx.fillStyle = "#FF0000";
  var ctx2 = canvas3.getContext("2d");
  ctx2.fillStyle = "#FF0000";

  setInterval(async () => {
    const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceExpressions()

    if(detections.length==1){
      for(var i=0; i<document.getElementsByClassName("winkScroll").length|0; i++) { document.getElementsByClassName("winkScroll")[i].style.backgroundColor = "#fed9ff"; }
    } else {
      for(var i=0; i<document.getElementsByClassName("winkScroll").length|0; i++) { document.getElementsByClassName("winkScroll")[i].style.backgroundColor = "white"; }
    }

    const resizedDetections = faceapi.resizeResults(detections, displaySize)
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)

    if(DEBUG) {
      faceapi.draw.drawDetections(canvas, resizedDetections)
      faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)
      faceapi.draw.drawFaceExpressions(canvas, resizedDetections)
    }

    canvas2.getContext('2d').clearRect(0, 0, canvas2.width, canvas2.height)
    canvas3.getContext('2d').clearRect(0, 0, canvas3.width, canvas3.height)

    if(detections.length==1){
      leftEye = detections[0].landmarks.getLeftEye()
      resizedLeftEye = resizedDetections[0].landmarks.getLeftEye()
    }

    var disX = distance(resizedLeftEye[0], resizedLeftEye[3]) /2
    var disY = distance(resizedLeftEye[1], resizedLeftEye[4]) -5


    // https://stackoverflow.com/questions/26015497/how-to-resize-then-crop-an-image-with-canvas
    ctx.drawImage( video,
      leftEye[0].x +10,        // start X
      leftEye[0].y - 3,        // start Y
      disX, disY,                                           // area to crop
      0, 0,                                                 // Place the result at 0, 0 in the canvas,
      canvas2.width, canvas2.height)                                             // with this width height (Scale)

      let src = cv.imread('canvas2');
      let dst = new cv.Mat();
      //bilateralFilter (https://docs.opencv.org/3.4/dd/d6a/tutorial_js_filtering.html)
      cv.cvtColor(src, src, cv.COLOR_RGBA2RGB, 0);
      cv.bilateralFilter(src, dst, 10, 15, 15);
      //erode (https://docs.opencv.org/3.4/d4/d76/tutorial_js_morphological_ops.html)
      let M = cv.Mat.ones(3, 3, cv.CV_8U);
      let anchor = new cv.Point(-1, -1);
      cv.erode(dst, src, M, anchor, 3, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
      //threshold (https://docs.opencv.org/3.4/d7/dd0/tutorial_js_thresholding.html)

      // https://docs.opencv.org/master/de/d06/tutorial_js_basic_ops.html
      let row = 3, col = 4;
      var A = 0;
      var rel_lum = 0;
      var lum2 = 0;
      var l709 = 0;
      var l601 = 0;
      if (src.isContinuous()) {
          let R = src.data[row * src.cols * src.channels() + col * src.channels()];
          let G = src.data[row * src.cols * src.channels() + col * src.channels() + 1];
          let B = src.data[row * src.cols * src.channels() + col * src.channels() + 2];
          A = src.data[row * src.cols * src.channels() + col * src.channels() + 3];
          rel_lum = (0.2126*R + 0.7152*G + 0.0722*B);
          lum2 = (0.299*R + 0.587*G + 0.114*B);
          l709 = 0.2126*R + 0.7152*G + 0.0722*B;
          l601 = 0.299*R + 0.587*G + 0.114*B;
      }

      if(A!=null && A!=0) {
        threshold = Math.floor(A/2) // using value A for calibration
      }
      cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
      cv.threshold(src, dst, threshold, 255, cv.THRESH_BINARY);

      if (dst.isContinuous()) {
          var BW = dst.data;
          console.log("No of 0: " + BW.filter(v => v === 0).length );
          console.log("Percentage of 0: " +  BW.filter(v => v === 0).length / BW.length );
      }

      //find contours (https://docs.opencv.org/3.4/d5/daa/tutorial_js_contours_begin.html)
      let dst2 = cv.Mat.zeros(dst.rows, dst.cols, cv.CV_8UC3);
      let contours = new cv.MatVector();
      let hierarchy = new cv.Mat();
      cv.findContours(dst, contours, hierarchy, cv.RETR_TREE, cv.CHAIN_APPROX_NONE);

      for (let i = 0; i < contours.size(); ++i) {
          let color = new cv.Scalar(255,0,0);
          cv.drawContours(dst2, contours, i, color, 1, cv.LINE_8, hierarchy, 100);
      }

      // https://docs.opencv.org/3.4/dc/dcf/tutorial_js_contour_features.html
      // get centroid and print it
      let cnt = contours.get(0);
      let Moments = cv.moments(cnt, false);

      let cx = Moments.m10/Moments.m00
      let cy = Moments.m01/Moments.m00

      console.log("cx: "+cx+"cy: "+cy)

      cv.imshow('canvas3', dst2);
      src.delete(); dst.delete(); contours.delete(); hierarchy.delete(); dst2.delete();

      // fill the buffer if centroid exists
      if( cx!=null && cx!=0 && cy!=null && cy!=0 && !Number.isNaN(cx) && !Number.isNaN(cy) ) {
        if(flag>bufferSize) {
          flag = 0
        }
        bufferX[flag] = cx
        bufferY[flag] = cy
        flag += 1
      }

      cx = movingAVG()[0]
      cy = movingAVG()[1]

      ctx.fillRect(cx, cy, 5, 5);

      if(cy >  5 * 150 / 7) {
        if(DEBUG) {
          text.innerHTML = "Wink Scrolling"
          text.style.backgroundColor = "red"
        }
        pageScroll()
      } else {
        if(DEBUG){
          text.innerHTML = "Static"
          text.style.backgroundColor = "white"
        }
      }

      if(DEBUG && detections.length==1) {
        if(detections[0].expressions.neutral>0.7) {
          document.body.style.backgroundColor = "white";
        }
        else if(detections[0].expressions.happy>0.7){
          document.body.style.backgroundColor = "blue";
        }
        else if(detections[0].expressions.sad>0.7){
          document.body.style.backgroundColor = "grey";
        }
        else if(detections[0].expressions.angry>0.7){
          document.body.style.backgroundColor = "red";
        }
        else if(detections[0].expressions.disgusted>0.7){
          document.body.style.backgroundColor = "green";
        }
        else if(detections[0].expressions.fearful>0.7){
          document.body.style.backgroundColor = "lightblue";
        }
        else if(detections[0].expressions.surprised>0.7){
          document.body.style.backgroundColor = "yellow";
        }
      }
/*
    for (var i = 0; i < 6; i++){
      ctx.fillRect(resizedDetections[0].landmarks.getLeftEye()[i].x, resizedDetections[0].landmarks.getLeftEye()[i].y, 2, 2);
    }
*/
//ctx.fillRect(resizedDetections[0].landmarks.getLeftEye()[0].x, resizedDetections[0].landmarks.getLeftEye()[0].y, 2, 2);
//ctx.fillRect(resizedDetections[0].landmarks.getLeftEye()[3].x, resizedDetections[0].landmarks.getLeftEye()[3].y, 2, 2);

  }, 100)
})

function movingAVG() {
  var x_total = 0
  var y_total = 0
  var actual_size = 0
  for(var i=0; i<bufferSize; i++){
    if( bufferX[i]!=0 && bufferY[i]!=0 && !Number.isNaN(bufferX[i]) && !Number.isNaN(bufferX[i]) ){
      x_total += bufferX[i]
      y_total += bufferY[i]
      actual_size += 1
    }
  }
  return [x_total/actual_size, y_total/actual_size]
}

function pageScroll() {
  for(var i=0; i<document.getElementsByClassName("winkScroll").length|0; i++) { document.getElementsByClassName("winkScroll")[i].scrollTop += 10; }
}

function goTop() {
  for(var i=0; i<document.getElementsByClassName("winkScroll").length|0; i++) { document.getElementsByClassName("winkScroll")[i].scrollTop = 0; }
}

function distance(p1, p2) {
  /* Calculate the distance between 2 points

  Arguments:
      p1 (x, y): First point
      p2 (x, y): Second point
  */
  return Math.sqrt( Math.pow((p2.x - p1.x), 2)  +  Math.pow((p2.y - p1.y), 2) );
}

function _middle_point(p1, p2) {
    /* Returns the middle point (x,y) between two points

    Arguments:
        p1 (x, y): First point
        p2 (x, y): Second point
    */
    x = int((p1.x + p2.x) / 2)
    y = int((p1.y + p2.y) / 2)
    return (x, y)
}

function getLandmarks(d){
  /* Returns the Landmarks

  Arguments:
      d (e.g. detections[0]): a face detection object
  */
  return d.landmarks.positions;
}

function getDetectionBox(d){
  /* Returns the detectionBox array which consists of bottomLeft, bottomRight, topLeft, topRight

  Arguments:
      d (e.g. detections[0]): a face detection object
  */
  return d.detection.box;
}
