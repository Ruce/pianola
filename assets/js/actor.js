class Actor {
	static Player = new Actor('Player');
	static Bot = new Actor('Bot');
	static Model = new Actor('Model');
	
	constructor(name) {
		this.name = name;
	}
	
	toString() {
		return `Actor.${this.name}`;
	}
}