const musicGroups = [
  { name: "Soda stereo", genre: "rock" },
  { name: "Callejeros", genre: "rock" },
  { name: "Tan bionica", genre: "pop" },
  { name: "Miranda", genre: "pop" },
  { name: "Los Palmeras", genre: "cumbia" },
  { name: "La Nueva Luna", genre: "cumbia" },
  { name: "Bizarap", genre: "urbano" },
  { name: "Duki", genre: "urbano" },
];

const genres = ["rock", "pop", "cumbia", "urbano"];
const genreToVector = genre => genres.map(g => g === genre ? 1 : 0);

const form = document.getElementById("ratingForm");
musicGroups.forEach((group, i) => {
  const div = document.createElement("div");
  div.className = "group";
  div.innerHTML = `
    <label for="group-${i}">${group.name}</label>
    <input type="number" id="group-${i}" min="1" max="10" required>
  `;
  form.appendChild(div);
});

document.getElementById("processBtn").addEventListener("click", async () => {
  const xs = [];
  const ys = [];

  musicGroups.forEach((group, i) => {
    const rating = parseFloat(document.getElementById(`group-${i}`).value);
    if (!isNaN(rating)) {
      xs.push(genreToVector(group.genre));
      ys.push(rating);
    }
  });

  if (xs.length < 2) {
    alert("Por favor calificá al menos 2 grupos.");
    return;
  }

  const inputTensor = tf.tensor2d(xs);
  const outputTensor = tf.tensor2d(ys, [ys.length, 1]);

  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [4], units: 4, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({ loss: 'meanSquaredError', optimizer: 'adam' });

  await model.fit(inputTensor, outputTensor, {
    epochs: 100,
    shuffle: true,
    verbose: 0
  });

  const predictions = genres.map(g => {
    const input = tf.tensor2d([genreToVector(g)]);
    const prediction = model.predict(input).dataSync()[0];
    return { genre: g, score: Math.max(1, Math.min(10, prediction)) }; // clamp entre 1 y 10
  });

  predictions.sort((a, b) => b.score - a.score);

  const resultsDiv = document.getElementById("results");
  resultsDiv.innerHTML = "<h2>Ranking según tus gustos:</h2><ol>" +
    predictions.map(p => `<li>${p.genre} (score: ${p.score.toFixed(2)})</li>`).join("") +
    "</ol>";
});
