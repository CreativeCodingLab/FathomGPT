<script lang="ts">
	import '../app.css';
</script>

<header>
	<h1>OVAI GPT Prototype</h1>
</header>
<slot />

<style>
	header {
		width: 100%;
		top: 0;
		display: grid;
		grid-template-columns: 20% 80%;
		min-height: 4rem;
		background-color: var(--accent-light);
	}

	h1 {
		grid-area: 1 / 1 / 2 / 2;
		place-self: center;
	}
</style>
