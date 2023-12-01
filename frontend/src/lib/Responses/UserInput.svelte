<script lang="ts">
	export let text: string;
</script>

<main>
	<h3>User said:</h3>
	<p>{text}</p>
</main>

<style>
	main {
		width: 100%;
		border-radius: 0.5rem;
		padding: 0.5rem;
		background-color: var(--mid);
		min-height: 3rem;
		color: white;
		display: grid;
		place-items: center end;
		gap: 0.5rem;
	}
</style>
