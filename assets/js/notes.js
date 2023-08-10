class NotesCanvas {
	constructor(canvasId, piano) {
		this.piano = piano;
		this.canvas = document.getElementById(canvasId);
		this.draw();
	}
	
	draw() {
		const ctx = this.canvas.getContext('2d');
		this.canvas.width = window.innerWidth;
		this.canvas.height = window.innerHeight - piano.canvas.height;
		ctx.fillStyle = '#222222';
		ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
	}
}