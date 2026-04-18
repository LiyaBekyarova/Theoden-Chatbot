fetch('/backgrounds')
  .then(r => r.json())
  .then(({ images }) => {
    if (images.length === 0) return;
    const pick = images[Math.floor(Math.random() * images.length)];
    const bg = document.getElementById('bg');
    bg.style.backgroundImage = `url('/static/images/backgrounds/${pick}')`;
  });

const chatWindow = document.querySelector('.chat-window');
const input      = document.querySelector('.input-area__field');
const sendBtn    = document.querySelector('.input-area__btn');

function addMessage(sender, role, text) {
  const message = document.createElement('div');
  message.className = `message message--${role}`;

  const avatar = document.createElement('img');
  avatar.className = 'message__avatar';
  avatar.src = role === 'bot' ? '/static/images/theoden.jpg' : '/static/images/user.svg';
  avatar.alt = sender;

  const content = document.createElement('div');
  content.className = 'message__content';

  const senderEl = document.createElement('div');
  senderEl.className = `message__sender message__sender--${role}`;
  senderEl.textContent = sender;

  const bubble = document.createElement('div');
  bubble.className = 'message__bubble';
  bubble.textContent = text;

  content.appendChild(senderEl);
  content.appendChild(bubble);
  message.appendChild(avatar);
  message.appendChild(content);
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
