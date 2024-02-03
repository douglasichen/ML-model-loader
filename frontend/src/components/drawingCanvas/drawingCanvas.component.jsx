import { useRef, useEffect, useState } from 'react';
import axios from 'axios';
import * as tf from "@tensorflow/tfjs"


// set model
// const modelURL = `${process.env.REACT_APP_URL}/converted/model/model.json`;
const modelURL = `${process.env.REACT_APP_URL}/converted4/model.json`;
console.log(modelURL)
const model = await tf.loadLayersModel(modelURL)
const modelInputShape = [28, 28];



// const ds = await MNISTDataset.create()
// console.log(ds);

const dataURL = `${process.env.REACT_APP_URL}/MNIST data/mnist_test.csv`;
// const dataURL = "https://pjreddie.com/media/files/mnist_test.csv"
//https://pjreddie.com/media/files/mnist_train.csv
const data = tf.data.csv(
	dataURL, {
		columnConfigs: {
			label: {
				isLabel: true
			}
		}
	});

const numOfFeatures = (await data.columnNames()).length - 1;

// Prepare the Dataset for training.
const flattenedDataset = data.map(({xs, ys}) => {
	// Convert xs(features) and ys(labels) from object form (keyed by
	// column name) to array form.
	return {xs:Object.values(xs), ys:Object.values(ys)};
})
//.batch(10);

const it = await flattenedDataset.iterator()
const xs = []
const ys = []
// read only the data for the first 5 rows
// all the data need not to be read once 
// since it will consume a lot of memory
const N=10000
for (let i = 0; i < N; i++) {
	let e = await it.next()
	xs.push(e.value.xs)
	ys.push(e.value.ys)
}
const features = tf.tensor(xs)
const labels = tf.tensor(ys)

const FB = tf.buffer(features.shape, features.dtype, features.dataSync());
const newFB = tf.buffer([N,28,28], features.dtype);
for (let t=0; t<N; t++) {
	for (let x=0; x<28; x++) {
		for (let y=0; y<28; y++) {
			let val = FB.get(t,y*28+x)/255.0
			newFB.set(val, t, y, x)
		}
	}
}

const LB = tf.buffer(labels.shape, labels.dtype, labels.dataSync())
const newLB = tf.buffer([N, 10], labels.dtype)
for (let t=0; t<N; t++) {
	for (let i=0; i<10; i++) {
		let val = (i+1==LB.get(t,0) ? 1 : 0)
		newLB.set(val, t, i)
	}
}

const newFeatures = newFB.toTensor()
const newLabels = newLB.toTensor()
// console.log(newFeatures)
// console.log(newLabels)

const optimizer = tf.train.rmsprop(0.00000001);
model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

const argMax = (b) => {
	let bestV = b.get(0, 0);
	let bestI = 0;
	for (let i = 1; i < b.shape[1]; i++) {
		if (b.get(0, i) > bestV) {
			bestV = b.get(0, i);
			bestI = i;
		}
	}
	return bestI;
}

const A = model.predict(newFeatures)
const AB = tf.buffer(A.shape, A.dtype, A.dataSync())
let gd=0
for (let t=0; t<N; t++) {
	let ans=0
	let bestVal=AB.get(t,0)
	for (let i=0; i<10; i++) {
		let val = AB.get(t,i)
		if (val>bestVal) {
			ans=i
			bestVal=val
		}
	}
	if (ans==LB.get(t)) {
		gd++
	}
}
const acc=gd/N
console.log(gd)
console.log(N)
console.log(acc)

const DrawingCanvas = () => {

	const displayCanvasRef = useRef(null)

	const canvasRef = useRef(null)
	const contextRef = useRef(null)
	const [isDrawing, setIsDrawing] = useState(false)
	const [response, setResponse] = useState(null)

	const borderThickness = 1;
	const canvas2Window = 0.5;
	const lineWidth = 40;

	// test model
	// model.evaluate(x, y);




	useEffect(() => {
		const canvas = canvasRef.current;
		canvas.width = Math.min(window.innerWidth, window.innerHeight) * 2 * canvas2Window;
		canvas.height = canvas.width;
		canvas.style.width = `${canvas.width / 2}px`;
		canvas.style.height = `${canvas.height / 2}px`;
		canvas.style.border = `${borderThickness}px solid black`;

		const context = canvas.getContext("2d");
		context.scale(2, 2);
		context.lineCap = "round";
		context.strokeStyle = "black";
		context.lineWidth = lineWidth;
		context.rect(0, 0, canvas.width, canvas.height);
		context.fillStyle = "white";
		context.fill();
		contextRef.current = context;


	}, [])
	const startDrawing = ({ nativeEvent }) => {
		const { offsetX, offsetY } = nativeEvent;
		contextRef.current.beginPath();
		contextRef.current.moveTo(offsetX, offsetY);
		setIsDrawing(true);
	}
	const finishDrawing = () => {
		contextRef.current.closePath();
		setIsDrawing(false);
	}
	const draw = ({ nativeEvent }) => {
		if (!isDrawing) {
			return;
		}
		const { offsetX, offsetY } = nativeEvent;
		contextRef.current.lineTo(offsetX, offsetY);
		contextRef.current.stroke();
	}
	const clearCanvas = () => {
		contextRef.current.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
		contextRef.current.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
		contextRef.current.rect(0, 0, canvasRef.current.width, canvasRef.current.height);
		contextRef.current.fillStyle = "white";
		contextRef.current.fill();

		contextRef.current.rect(0, 0, canvasRef.current.width, canvasRef.current.height);
		contextRef.current.fillStyle = "white";
		contextRef.current.fill();
	}


	const predict = async () => {
		const context = contextRef.current;
		const canvas = canvasRef.current;
		const imageData = context.getImageData(0, 0, canvas.width, canvas.height);

		// gray scale + resize
		var imageTensor = tf.browser.fromPixels(imageData, 3);
		// console.log(imageTensor)
		imageTensor = tf.image.resizeBilinear(imageTensor, modelInputShape);
		imageTensor = tf.image.rgbToGrayscale(imageTensor);

		
		// imageTensor.print()

		// set to: 1 - a/255 (0-1 domain and inversion)
		var buffer = tf.buffer(imageTensor.shape, imageTensor.dtype, imageTensor.dataSync());
		var newBuffer = tf.buffer([1, modelInputShape[0], modelInputShape[1]], imageTensor.dtype)
		var newBuffer2 = tf.buffer([modelInputShape[0], modelInputShape[1], 1], imageTensor.dtype)
		for (let y = 0; y < modelInputShape[0]; y++) {
			for (let x = 0; x < modelInputShape[1]; x++) {
				newBuffer.set(1 - buffer.get(y, x, 0) / 255.0, 0, y, x);
				newBuffer2.set(1 - buffer.get(y, x, 0) / 255.0, y, x, 0)
			}
		}
		// console.log(newBuffer);
		imageTensor = newBuffer.toTensor();
		var imageTensor2 = newBuffer2.toTensor();

		imageTensor2 = tf.image.resizeNearestNeighbor(imageTensor2, [280, 280])
		
		await tf.browser.toPixels(imageTensor2, displayCanvasRef.current);
		// imageTensor.print();

		// feed into network
		var res = model.predict(imageTensor)
		res = tf.buffer([1, 10], res.dtype, res.dataSync());

		

		await setResponse(argMax(res));
	};

	return (
		<div>
			<div>
				<canvas
					id="DrawingCanvas"
					onMouseDown={startDrawing}
					onMouseUp={finishDrawing}
					onMouseMove={draw}
					ref={canvasRef}
				/>
			</div>
			<div>
				<canvas
					id="DisplayCanvas"
					ref={displayCanvasRef}
				/>
			</div>
			<div>
				<button type="submit" text-align="center" onClick={clearCanvas}>Clear</button>
			</div>
			<div>
				<button type="submit" text-align="center" onClick={predict}>Predict</button>
			</div>
			<div>
				<h1 id="PredictText">Prediction: {response}</h1>
			</div>
		</div>

	)
}

export default DrawingCanvas;
