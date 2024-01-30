<script lang="ts">
	import simpleheat from 'simpleheat';
	import { onMount } from 'svelte';

	export let mapBoxKey =
		'pk.eyJ1IjoidGNsdWZmIiwiYSI6ImNsbWpoNGJ3MTAzYm8ycXJ4ZDVieTk3ODYifQ.__pspVfdjrgiM_ACd5jhdg';

	export let responseText: string;
	export let positionData: {x: number; y: number; z: number}[];

	let heatMapCanvas: HTMLCanvasElement;
	// minimal heatmap instance configuration
	onMount(() => {
		// heatmap data format
		let heat = simpleheat(heatMapCanvas);
		// the +- 0.1 is to add margins between the map edge and data
		let maxLat = Math.max(...positionData.map((obj) => obj.x)) + 0.1;
		let minLat = Math.min(...positionData.map((obj) => obj.x)) - 0.1;
		let maxLong = Math.max(...positionData.map((obj) => obj.y)) + 0.1;
		let minLong = Math.min(...positionData.map((obj) => obj.y)) - 0.1;
		console.log("Max X, Min X, Max Y, Min Y");
		console.log(maxLong, minLong, maxLat, minLat);


		let ratio = (maxLat - minLat) / (maxLong - minLong);
		let xScale = 880
		let yScale = Math.ceil(xScale * ratio);

		let mapWidth = xScale;
		heatMapCanvas.width = mapWidth;
		let mapHeight = yScale;
		heatMapCanvas.height = mapHeight;

		let imageURL = `https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/[${minLong},${minLat},${maxLong},${maxLat}]/${xScale}x${yScale}?access_token=${mapBoxKey}`;

		let arrayOfArrays: [number,number,number][];

		arrayOfArrays = positionData.map((obj) => [
			remap(obj.x, minLat, maxLat, 0, xScale), //WEIRD SHIFTING, FIX LATER
			remap(obj.y, minLong, maxLong, 0, yScale), //WEIRD SHIFTING, FIX LATER
			obj.z
		]);
		
		

		//console.table(positionData);

		// radius, blur amount
		heat.radius(2,5);
		heat.resize();

		heat.data(arrayOfArrays);
		heat.draw();

		heatMapCanvas.style.backgroundImage = `url(${imageURL})`;

		window.scrollTo(0, document.body.scrollHeight);
	});

	function remap(value: number, from1: number, to1: number, from2: number, to2: number): number {
		return ((value - from1) / (to1 - from1)) * (to2 - from2) + from2;
	}
</script>

<main class="fathomChatContainer">
	<p>{responseText}</p>
	<br />
	<p class="dev-note">Developer Note: background map image is NOT perfectly lined up with the plotted data YET</p>
	<canvas bind:this={heatMapCanvas} />
	
</main>

<style>

	canvas {
		border-radius: 0.25rem;
		background-repeat: no-repeat;
		background-position: center;
		background-size: cover;
	}
	.dev-note {
		font-style: italic;
		color: red;
	}
</style>
