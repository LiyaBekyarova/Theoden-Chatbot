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

function showTyping() {
  const el = document.createElement('div');
  el.className = 'typing-indicator';
  el.id = 'typing';

  const avatar = document.createElement('img');
  avatar.className = 'typing-indicator__avatar';
  avatar.src = '/static/images/theoden.jpg';
  avatar.alt = 'Théoden';

  const dots = document.createElement('div');
  dots.className = 'typing-indicator__dots';
  dots.innerHTML = '<span></span><span></span><span></span>';

  el.appendChild(avatar);
  el.appendChild(dots);
  chatWindow.appendChild(el);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function hideTyping() {
  const el = document.getElementById('typing');
  if (el) el.remove();
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;

  addMessage('You (Rider of the Mark)', 'user', text);
  input.value = '';
  sendBtn.disabled = true;

  showTyping();

  const res  = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });

  hideTyping();
  sendBtn.disabled = false;

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
