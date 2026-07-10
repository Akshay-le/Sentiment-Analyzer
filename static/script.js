const reviewEl = document.getElementById('review');
const charCount = document.getElementById('charCount');
const analyzeBtn = document.getElementById('analyzeBtn');
const sampleBtn = document.getElementById('sampleBtn');
const errorMsg = document.getElementById('errorMsg');

const ticketEmpty = document.getElementById('ticketEmpty');
const ticketResult = document.getElementById('ticketResult');
const ticketSerial = document.getElementById('ticketSerial');
const verdictLabel = document.getElementById('verdictLabel');
const verdictIcon = document.getElementById('verdictIcon');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceFill = document.getElementById('confidenceFill');
const posFill = document.getElementById('posFill');
const negFill = document.getElementById('negFill');
const posPct = document.getElementById('posPct');
const negPct = document.getElementById('negPct');
const keywordChips = document.getElementById('keywordChips');

const SAMPLES = [
  "This film absolutely blew me away. The performances were nuanced, the cinematography was stunning, and the score elevated every scene. I left the theater speechless — easily one of the best movies I've seen this decade.",
  "I really wanted to like this one but it was a slog. The pacing drags for the first hour, the dialogue feels wooden, and the ending makes almost no sense given everything that came before it. Disappointing.",
  "A solid, well-crafted thriller with a genuinely surprising third act. A few side characters feel underwritten, but the lead performance carries the whole thing. Worth a watch."
];

let serial = 100132;

function setLoading(isLoading){
  analyzeBtn.disabled = isLoading;
  analyzeBtn.classList.toggle('loading', isLoading);
}

function showError(msg){
  errorMsg.textContent = msg || '';
}

reviewEl.addEventListener('input', () => {
  charCount.textContent = `${reviewEl.value.length} characters`;
});

sampleBtn.addEventListener('click', () => {
  const pick = SAMPLES[Math.floor(Math.random() * SAMPLES.length)];
  reviewEl.value = pick;
  charCount.textContent = `${pick.length} characters`;
  showError('');
  reviewEl.focus();
});

async function analyze(){
  const text = reviewEl.value.trim();
  showError('');

  if (!text){
    showError('Please enter a movie review first.');
    return;
  }

  setLoading(true);
  try{
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({review: text})
    });
    const data = await res.json();

    if (!res.ok){
      showError(data.error || 'Something went wrong. Please try again.');
      return;
    }

    renderResult(data);
  }catch(err){
    showError('Could not reach the analyzer. Check your connection and try again.');
  }finally{
    setLoading(false);
  }
}

function renderResult(data){
  ticketEmpty.hidden = true;
  ticketResult.hidden = false;

  serial += 1;
  ticketSerial.textContent = `NO. ${String(serial).padStart(6, '0')}`;

  const isPositive = data.label === 'positive';
  verdictLabel.textContent = isPositive ? 'POSITIVE' : 'NEGATIVE';
  verdictLabel.className = 'verdict-label ' + (isPositive ? 'is-positive' : 'is-negative');
  verdictIcon.textContent = isPositive ? '\u2605' : '\u2716';
  verdictIcon.style.color = isPositive ? 'var(--gold)' : 'var(--velvet-bright)';

  confidenceValue.textContent = `${data.confidence}%`;
  requestAnimationFrame(() => {
    confidenceFill.style.width = `${data.confidence}%`;
    posFill.style.width = `${data.positive_prob}%`;
    negFill.style.width = `${data.negative_prob}%`;
  });
  posPct.textContent = `${data.positive_prob}%`;
  negPct.textContent = `${data.negative_prob}%`;

  keywordChips.innerHTML = '';
  if (data.top_words && data.top_words.length){
    data.top_words.forEach(w => {
      const chip = document.createElement('span');
      chip.className = `chip ${w.direction}`;
      chip.textContent = w.word;
      keywordChips.appendChild(chip);
    });
  }else{
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = 'no strong signal words found';
    keywordChips.appendChild(chip);
  }
}

analyzeBtn.addEventListener('click', analyze);
reviewEl.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter'){
    analyze();
  }
});
