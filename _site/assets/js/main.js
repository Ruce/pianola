class Piano {
	constructor(canvasId, octaves) {
  	this.octaves = octaves;
    this.whiteKeys = 7 * this.octaves + 3; // 3 additional keys before and after main octaves
		this.blackKeys = 5 * this.octaves + 1; // 1 additional key in the 0th octave
    
    this.canvas = document.getElementById(canvasId);
		this.ctx = this.canvas.getContext('2d');
    this.drawKeyboard();
    this.canvas.addEventListener('mousedown', this.keyboardClicked.bind(this));
    //this.canvas.addEventListener('mousemove', this.mouseMoveKeyboard.bind(this));
  }
  
  static get blackKeyWidthRatio() {
  	return 1/2;
  }
  
	static get blackKeyHeightRatio() {
  	return 2/3;
  }
  
  // Key number of the white keys relative to an octave
  static get whiteKeyNumbers() {
  	return [1, 3, 5, 6, 8, 10, 12];
  }
  
  // Key number of the black keys relative to an octave
  static get blackKeyNumbers() {
  	return [2, 4, 7, 9, 11];
  }
  
  // Top left coordinate of each black key relative to start of an octave (normalised by whiteKeyWidth)
  static get blackKeyPos() {
  	return [
      2/3,
      1 + 5/6,
      3 + 5/8,
      4 + 3/4,
      5 + 7/8
    ];
  }
  
  getKeyByCoord(x, y) {
  	const octaveWidth = this.whiteKeyWidth * 7;
    const o = Math.floor((x + (this.whiteKeyWidth * 5)) / octaveWidth); // Current octave
    const deltaX = x - ((o-1) * octaveWidth) - (2 * this.whiteKeyWidth); // x position relative to octave
    
  	if (y > this.blackKeyHeight) {
    	// Must be a white key
      const n = Math.floor(deltaX / this.whiteKeyWidth);
      const keyNum = Piano.whiteKeyNumbers[n] + o*12 - 9;
      return keyNum;
    } else if (o === this.octaves + 1) {
      // Only highest C is in the highest octave
      return this.octaves * 12 + 4;
    } else {
      for (let i=0; i < Piano.blackKeyPos.length; i++) {
      	if (o === 0 && i < 4) {
        	// 0th octave does not have first 4 black keys
          continue;
        }
        const pos = Piano.blackKeyPos[i];
        const blackKeyLeft = this.whiteKeyWidth * pos;
        const blackKeyRight = blackKeyLeft + this.blackKeyWidth;
        // Except for octave 0, which only has 1 black key
        if (deltaX >= blackKeyLeft && deltaX <= blackKeyRight) {
          const keyNum = Piano.blackKeyNumbers[i] + o*12 - 9;
          return keyNum;
        }
      }
      // Not a black key, therefore must be a white key
      const n = Math.floor(deltaX / this.whiteKeyWidth);
      const keyNum = Piano.whiteKeyNumbers[n] + o*12 - 9;
      return keyNum;
    }
  }
  
  mouseMoveKeyboard(event) {
  	console.log(this.getKeyByCoord(event.clientX, event.clientY));
  }
  
  keyboardClicked(event) {
    const canvasRect = this.canvas.getBoundingClientRect();
    const clickX = event.clientX - canvasRect.left;
    const clickY = event.clientY - canvasRect.top;
    console.log(this.getKeyByCoord(clickX, clickY));
  }
  
  drawKeyboard() {
    const ctx = this.ctx;
    ctx.canvas.width = window.innerWidth;
    ctx.canvas.height = ctx.canvas.width / 7;

    this.whiteKeyWidth = ctx.canvas.width / this.whiteKeys;
    this.whiteKeyHeight = ctx.canvas.height;
    this.blackKeyWidth = this.whiteKeyWidth * Piano.blackKeyWidthRatio;
    this.blackKeyHeight = this.whiteKeyHeight * Piano.blackKeyHeightRatio;
    const [whiteKeyWidth, whiteKeyHeight, blackKeyWidth, blackKeyHeight] = [this.whiteKeyWidth, this.whiteKeyHeight, this.blackKeyWidth, this.blackKeyHeight];
    
    ctx.fillStyle = 'white';
    for (let i = 0; i < this.whiteKeys; i++) {
      const x = i * whiteKeyWidth;
      ctx.fillRect(x, 0, whiteKeyWidth, whiteKeyHeight);
      ctx.strokeRect(x, 0, whiteKeyWidth, whiteKeyHeight);
    }

    ctx.fillStyle = 'grey';
    for (let i = 0; i < this.blackKeys; i++) {
      const k = (i+4) % 5; // Index of the 5 black keys in `blackKeyPos`
      const o = Math.floor((i-1) / 5); // Current octave
      const x = whiteKeyWidth * (Piano.blackKeyPos[k] + o*7 + 2);
      ctx.fillRect(x, 0, blackKeyWidth, blackKeyHeight);
      ctx.strokeRect(x, 0, blackKeyWidth, blackKeyHeight);
    }
  }
}

document.addEventListener("DOMContentLoaded", initialisePiano);

const octaves = 5;
function initialisePiano() {
	const piano = new Piano('pianoCanvas', octaves);

	var resizeTimeout = false;
	const resizeDelay = 250;
	window.onresize = function () {
		clearTimeout(resizeTimeout);
	  resizeTimeout = setTimeout(piano.drawKeyboard.bind(piano), resizeDelay);
	}
}