const path = require('path')
const fs = require('fs')
const {
  fr,
  getAppdataPath,
  ensureAppdataDirExists
} = require('./commons')

fr.winKillProcessOnExit()
ensureAppdataDirExists()

const trainedModelFile = 'faceRecognition2Model_150.json'
const trainedModelFilePath = path.resolve(getAppdataPath(), trainedModelFile)

const dataPath = path.resolve('./examples/data')
const facesPath = path.resolve(dataPath, 'faces')
const classNames = ['hung', 'howard']

const detector = fr.FaceDetector()
const recognizer = fr.FaceRecognizer()

if (!fs.existsSync(trainedModelFilePath)) {
  console.log('%s not found, start training recognizer...', trainedModelFile)
  const allFiles = fs.readdirSync(facesPath)
  const imagesByClass = classNames.map(c =>
    allFiles
    .filter(f => f.includes(c))
    .map(f => path.join(facesPath, f))
    .map(fp => fr.loadImage(fp))
  )

  imagesByClass.forEach((faces, label) =>
    recognizer.addFaces(faces, classNames[label])
  )

  fs.writeFileSync(trainedModelFilePath, JSON.stringify(recognizer.serialize()));
} else {
  console.log('found %s, loading model', trainedModelFile)

  recognizer.load(require(trainedModelFilePath))

  console.log('imported the following descriptors:')
  console.log(recognizer.getDescriptorState())
}

const bbtThemeImgs = fs.readdirSync(dataPath)
  .filter(f => f.includes('bbt'))
  .map(f => path.join(dataPath, f))
  .map(fp => fr.loadImage(fp))