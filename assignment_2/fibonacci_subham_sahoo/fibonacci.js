
// const myFirstTensor = tf.scalar(42)
// console.log(myFirstTensor)
// myFirstTensor.print()


// const oneDimTensor = tf.tensor1d([1, 2, 3])
// oneDimTensor.print()

// Preparing the training data

function fibonacci(A, B, num){
    var a = A, b = B, temp;
    var seq = []

    while (num > 0){
        temp = a;
        a = a + b;
        b = temp;
        seq.push(b)
        num--;
    }
    return seq;
}

const fibs = fibonacci(15,10, 50)
console.log(fibs, fibs.length)
const xs = tf.tensor1d(fibs.slice(0, fibs.length - 1))
const ys = tf.tensor1d(fibs.slice(1))
xs.print()
ys.print()

const xmin = xs.min();
const xmax = xs.max();
const xrange = xmax.sub(xmin);

function norm(x) {
    return x.sub(xmin).div(xrange);
}

xsNorm = norm(xs)
ysNorm = norm(ys)

xsNorm.print()
ysNorm.print()
// Building our model

const a = tf.variable(tf.scalar(Math.random()))
const b = tf.variable(tf.scalar(Math.random()))

a.print()
b.print()

function predict(x) {
    return tf.tidy(() => {
        return a.mul(x).add(b)
    });
}

// Training

function loss(predictions, labels) {
    return predictions.sub(labels).square().mean();
}


const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

const numIterations = 100000;
const errors = []

for (let iter = 0; iter < numIterations; iter++) {
    optimizer.minimize(() => {
        const predsYs = predict(xsNorm);
        const e = loss(predsYs, ysNorm);
        errors.push(e.dataSync())
        return e
    });
}

// Making predictions

console.log(errors[0])
console.log(errors[numIterations - 1])

xTest = tf.tensor1d([15, 25])
predict(xTest).print()

a.print()
b.print()