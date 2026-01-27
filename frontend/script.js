async function askQuestion() {
  const question = document.getElementById("question").value;
  if (!question) return;

  const answerDiv = document.getElementById("answer");
  answerDiv.innerText = "Thinking...";

  try {
    const response = await fetch("http://127.0.0.1:8000/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        question: question
      })
    });

    const data = await response.json();

    // Show answer text
    answerDiv.innerText = data.answer;

    // Play WAV audio
    const audio = new Audio(data.audio_url);
    audio.play();

  } catch (err) {
    answerDiv.innerText = "Error connecting to backend.";
    console.error(err);
  }
}
