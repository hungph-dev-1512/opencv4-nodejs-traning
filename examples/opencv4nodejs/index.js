const cv = require('opencv4nodejs');
const path = require('path')
const fs = require('fs')
const fr = require('face-recognition').withCv(cv)
const {
  getAppdataPath
} = require('../commons')

const trainedModelFile = 'faceRecognition2Model_150.json'
const trainedModelFilePath = path.resolve(getAppdataPath(), trainedModelFile)
const recognizer = fr.FaceRecognizer()

if (!fs.existsSync(trainedModelFilePath)) {
  throw new Error('model file not found, please run the faceRecognition2 example first to train and save the model')
} else {
  recognizer.load(require(trainedModelFilePath))
}

function drawRectWithText(image, rect, text, color) {
  const thickness = 2
  image.drawRectangle(
    new cv.Point(rect.x, rect.y),
    new cv.Point(rect.x + rect.width, rect.y + rect.height),
    color,
    cv.LINE_8,
    thickness
  )

  const textOffsetY = rect.height + 20
  image.putText(
    text,
    new cv.Point(rect.x, rect.y + textOffsetY),
    cv.FONT_ITALIC,
    0.6,
    color,
    thickness
  )
}

grabFrames = (videoFile, delay, onFrame) => {
  const cap = new cv.VideoCapture(videoFile);
  let done = false;
  const intvl = setInterval(() => {
    let frame = cap.read();
    // loop back to start on end of stream reached
    if (frame.empty) {
      cap.reset();
      frame = cap.read();
    }
    onFrame(frame);

    const key = cv.waitKey(delay);
    done = key !== -1 && key !== 255;
    if (done) {
      clearInterval(intvl);
      console.log('Key pressed, exiting.');
    }
  }, 0);
};

runVideoFaceDetection = (src, detectFaces) => grabFrames(src, 1, (frame) => {
  console.time('detection time');
  const frameResized = frame.resizeToMax(800);

  // detect faces
  const faceRects = detectFaces(frameResized);
  if (faceRects.length) {
      // mark faces with distance > 0.6 as unknown
      const unknownThreshold = 0.6
      faceRects.forEach((faceRect) => {
        const { rect, face } = faceRect
        const cvFace = fr.CvImage(face)
        const prediction = recognizer.predict(cvFace, unknownThreshold)
        console.log(prediction)
        const text = `${prediction.className} (${prediction.distance})`
        const blue = new cv.Vec(255, 0, 0)
        // draw Rect
        drawRectWithText(frameResized, faceRect.rect, text, blue)
      })
  }
  cv.imshow('face detection', frameResized);
  // console.timeEnd('detection time');
  cv.waitKey(1)
});

const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);

const webcamPort = 0;

function detectFaces(img) {
  const options = {
    minSize: new cv.Size(100, 100),
    scaleFactor: 1.2,
    minNeighbors: 10,    
  }
  const { objects, map } = classifier.detectMultiScale(img.bgrToGray(), options)

  return objects
  .map(rect => ({      
    rect,
    face: img.getRegion(rect)
  }))
}
runVideoFaceDetection(webcamPort, detectFaces);
cv.waitKey()
