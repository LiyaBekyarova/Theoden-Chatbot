const chatWindow = document.querySelector('.chat-window');
const input      = document.querySelector('.input-area__field');
const sendBtn    = document.querySelector('.input-area__btn');

function addMessage(sender, role, text) {
  const message = document.createElement('div');
  message.className = 'message';

  const senderEl = document.createElement('div');
  senderEl.className = `message__sender message__sender--${role}`;
  senderEl.textContent = sender;

  const bubble = document.createElement('div');
  bubble.className = 'message__bubble';
  bubble.textContent = text;

  message.appendChild(senderEl);
  message.appendChild(bubble);
  chatWindow.appendChild(message);

  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;

  addMessage('You (Rider of the Mark)', 'user', text);
  input.value = '';

  const res  = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });

  const data = await res.json();
  addMessage(`${data.sender}, King of Rohan`, 'bot', data.response);
}

sendBtn.addEventListener('click', sendMessage);
input.addEventListener('keydown', e => {
  if (e.key === 'Enter') sendMessage();
});

addMessage(
  'Théoden, King of Rohan',
  'bot',
  'Hail, friend of Rohan!\n\nI am Théoden, son of Thengel, King of the Mark.\nSpeak thy mind, and let us hold counsel together.\nThe hearth is warm and the mead awaits!'
);
