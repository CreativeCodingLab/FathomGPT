<script lang="ts">
	export let text: string;
</script>

<main>
	<h3>Fathom said:</h3>
	<br/>
	<p>{text}</p>
</main>

<style>
	main {
		width: 100%;
		border-radius: 0.5rem;
		padding: 0.5rem;
		background-color: var(--background-light);
		min-height: 5rem;
		color: white;
		display: grid;
		place-items: center start;
	}
</style>
