class PianoMode {
	static Composer = new PianoMode('composer');
	static Autoplay = new PianoMode('autoplay');
	static Freeplay = new PianoMode('freeplay');
	
	static getModeByName(name) {
		switch (name) {
			case 'composer':
				return PianoMode.Composer;
			case 'autoplay':
				return PianoMode.Autoplay;
			case 'freeplay':
				return PianoMode.Freeplay;
		}
	}
				
	constructor(name) {
		this.name = name;
	}
	
	toString() {
		return `PianoMode.${this.name}`;
	}
}