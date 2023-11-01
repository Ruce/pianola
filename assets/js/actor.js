class Actor {
	static Player = new Actor('player');
	static Bot = new Actor('bot');
	static Model = new Actor('model');
	
	constructor(name) {
		this.name = name;
	}
	
	toString() {
		return `Actor.${this.name}`;
	}
}