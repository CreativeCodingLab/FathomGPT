<script lang="ts">
	export let text: string;
	export let image: string | null;
</script>

<main>
	<p>{text}</p>
	{#if image!=null}
		<img src="{image}"/>
	{/if}
</main>

<style>
	main {
		border-radius: var(--chat-bubble-radius);
		padding: var(--chat-padding);
		background-color: var(--mid);
		min-height: 3rem;
		gap: 0.5rem;
		width: fit-content;
		display: flex;
		align-self: end;
		flex-direction: column;
	}
	main *{
		color: white;
	}
	img{
		max-width: min(100%, 500px);
		max-height: 500px;
		width: auto;
		height: auto;
		padding: 10px 0px;
	}
</style>
