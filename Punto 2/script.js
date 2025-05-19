const cars = [
  { Miles_per_Gallon: 18, Horsepower: 130 },
  { Miles_per_Gallon: 15, Horsepower: 165 },
  { Miles_per_Gallon: 18, Horsepower: 150 },
  { Miles_per_Gallon: 16, Horsepower: 150 },
  { Miles_per_Gallon: 17, Horsepower: 140 },
  { Miles_per_Gallon: 15, Horsepower: 198 },
  { Miles_per_Gallon: 14, Horsepower: 220 },
  { Miles_per_Gallon: 14, Horsepower: 215 },
  { Miles_per_Gallon: 14, Horsepower: 225 },
  { Miles_per_Gallon: 15, Horsepower: 190 },
];
function convertToTensor(data) {
  return tf.tidy(() => {
    tf.util.shuffle(data);

    const inputs = data.map(d => d.horsepower);
    const labels = data.map(d => d.mpg);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();  
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
  });  
}


function plotData() {
  const values = cars.map(car => ({
    x: car.Horsepower,
    y: car.Miles_per_Gallon
  }));

  tfvis.render.scatterplot(
    { name: 'Caballos de fuerza vs. MPG' },
    { values },
    {
      xLabel: 'Caballos de fuerza',
      yLabel: 'Millas por galón',
      height: 300
    }
  );
}

document.addEventListener('DOMContentLoaded', plotData);

function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
  model.add(tf.layers.dense({ units: 1, useBias: true }));

  return model;
}

const model = createModel();
tfvis.show.modelSummary({ name: 'Resumen del modelo' }, model);

async function trainModel(model, inputs, labels) {
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse']
  });

  const surface = { name: 'Historial de entrenamiento', tab: 'Entrenamiento' };

  const history = await model.fit(inputs, labels, {
    batchSize: 5,
    epochs: 50,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(surface, ['loss', 'mse'], { height: 200, callbacks: ['onEpochEnd'] })
  });

  return history;
}

function testModel(model, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100);
    const preds = model.predict(xs.reshape([100, 1]));

    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] };
  });

  const originalPoints = inputData.map(d => ({
    x: d.Horsepower,
    y: d.Miles_per_Gallon
  }));

  tfvis.render.scatterplot(
    { name: 'Predicciones vs Datos Originales' },
    { values: [originalPoints, predictedPoints], series: ['originales', 'predicciones'] },
    {
      xLabel: 'Caballos de fuerza',
      yLabel: 'Millas por galón',
      height: 300
    }
  );
}
async function run() {
  const data = cars;
  const values = data.map(d => ({
    x: d.Horsepower,
    y: d.Miles_per_Gallon
  }));
  tfvis.render.scatterplot(
    { name: 'Datos' },
    { values },
    {
      xLabel: 'Caballos de fuerza',
      yLabel: 'Millas por galón',
      height: 300
    }
  );

  const model = createModel();
  tfvis.show.modelSummary({ name: 'Modelo' }, model);

  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;

  await trainModel(model, inputs, labels);
  console.log("Entrenamiento finalizado");

  testModel(model, data, tensorData);
}

document.addEventListener('DOMContentLoaded', run);
