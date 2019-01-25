const cv = require("opencv4nodejs");
const path = require("path");
const fs = require("fs");
const fr = require("face-recognition").withCv(cv);
const {
  getAppdataPath
} = require("../commons");

const trainedModelFile = "faceRecognition2Model_150.json";
const trainedModelFilePath = path.resolve(getAppdataPath(), trainedModelFile);
const recognizer = fr.FaceRecognizer();

if (!fs.existsSync(trainedModelFilePath)) {
  throw new Error(
    "model file not found, please run the faceRecognition2 example first to train and save the model"
  );
} else {
  recognizer.load(require(trainedModelFilePath));
}

function drawRectWithText(image, rect, text, color) {
  const thickness = 2;
  image.drawRectangle(
    new cv.Point(rect.x, rect.y),
    new cv.Point(rect.x + rect.width, rect.y + rect.height),
    color,
    cv.LINE_8,
    thickness
  );

  const textOffsetY = rect.height + 20;
  image.putText(
    text,
    new cv.Point(rect.x, rect.y + textOffsetY),
    cv.FONT_ITALIC,
    0.6,
    color,
    thickness
  );
}

grabFrames = (videoFile, delay, onFrame) => {
  // const cap = new cv.VideoCapture(0);
  // open capture from webcam
  console.log("[Notification] Start Streaming Face Recognition \n");
  console.log(
    "**********************************************************************************"
  );
  console.log(
    "* ID | Human Name |           Time Start           |           Time End          *"
  );
  const wCap = new cv.VideoCapture(videoFile);
  let done = false;
  const intvl = setInterval(() => {
    let frame = wCap.read();
    // loop back to start on end of stream reached
    if (frame.empty) {
      wCap.reset();
      frame = wCap.read();
    }
    onFrame(frame);

    const key = cv.waitKey(delay);
    done = key !== -1 && key !== 255;
    if (done) {
      clearInterval(intvl);
      console.log(
        "********************************************************************************** \n"
      );
      console.log("[Notification] End Streaming Face Recognition \n");
      console.log("[Event] Key pressed, exiting.");
    }
  }, 0);
};


// global.flagsTimestamp = new Array();
// flagsTimestamp["hung"] = 0;
// flagsTimestamp["howard"] = 0;
// global.facesCount = 0;
// global.timeIn = 0;
// global.timeOut = 0;
global.timeIn = new Array();
timeIn["hung"] = 0;
timeIn["howard"] = 0;
global.timeOut = new Array();
timeOut["hung"] = 0;
timeOut["howard"] = 0;

runVideoFaceDetection = (src, detectFaces) =>
  grabFrames(src, 10, frame => {
    global.flagsDetect = new Array();
    flagsDetect["hung"] = 0;
    flagsDetect["howard"] = 0;

    // console.time("detection time");
    const frameResized = frame.resizeToMax(800);
    // detect faces
    const faceRects = detectFaces(frameResized);
    if (faceRects.length) {
      faceRects.forEach(faceRect => {
        const {
          rect,
          face
        } = faceRect;
        const cvFace = fr.CvImage(face);
        const prediction = recognizer.predictBest(cvFace);
        const text = `${prediction.className} (${prediction.distance})`;
        const blue = new cv.Vec(255, 0, 0);
        // draw Rect
        drawRectWithText(frameResized, faceRect.rect, text, blue);
        // log timestamp
        if (!timeIn[prediction.className]) {
          timeIn[prediction.className] = new Date();
        }
        if (timeOut[prediction.className]) {
          timeOut[prediction.className] = 0;
        }
        flagsDetect[prediction.className] = 1;
      });
    } else {
      if (timeOut['hung']) {
        let seconds = (new Date().getTime() - timeOut['hung'].getTime()) / 1000;
        if (seconds > 10) {
          if ((timeOut['hung'].getTime() - timeIn['hung'].getTime()) > 10) {
            console.log(
              "* ---|------------|--------------------------------|---------------------------- *"
            );
            console.log(
              "* 1  |    hung    |      " +
              timeIn['hung'].toISOString()
              .replace(/T/, " ")
              .replace(/\..+/, "") +
              "       |     " +
              timeOut['hung'].toISOString()
              .replace(/T/, " ")
              .replace(/\..+/, "") +
              "     *"
            );
            timeIn['hung'] = 0;
            timeOut['hung'] = 0;
          }
        }
      }
      if (timeOut['howard']) {
        let seconds = (new Date().getTime() - timeOut['howard'].getTime()) / 1000;
        if (seconds > 10) {
          if ((timeOut['howard'].getTime() - timeIn['howard'].getTime()) > 10) {
            console.log(
              "* ---|------------|--------------------------------|---------------------------- *"
            );
            console.log(
              "* 1  |   howard   |      " +
              timeIn['howard'].toISOString()
              .replace(/T/, " ")
              .replace(/\..+/, "") +
              "       |     " +
              timeOut['howard'].toISOString()
              .replace(/T/, " ")
              .replace(/\..+/, "") +
              "     *"
            );
            timeIn['howard'] = 0;
            timeOut['howard'] = 0;
          }
        }
      }
    }

    if (!flagsDetect["hung"] && timeIn["hung"] && !timeOut["hung"]) {
      timeOut["hung"] = new Date();
    }
    if (!flagsDetect["howard"] && timeIn["howard"] && !timeOut["howard"]) {
      timeOut["howard"] = new Date();
    }
    // });
    cv.imshow("face detection", frameResized);
    // console.timeEnd("detection time");
    cv.waitKey(1);
  });

const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);

const webcamPort = 0;

function detectFaces(img) {
  const options = {
    minSize: new cv.Size(100, 100),
    scaleFactor: 1.2,
    minNeighbors: 10
  };
  const {
    objects,
    map
  } = classifier.detectMultiScale(
    img.bgrToGray(),
    options
  );

  return objects.map(rect => ({
    rect,
    face: img.getRegion(rect).copy()
  }));
}

runVideoFaceDetection(webcamPort, detectFaces);
cv.waitKey();