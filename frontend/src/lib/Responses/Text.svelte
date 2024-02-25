<script lang="ts">
	export let text: string;

	if(typeof text === "object"){
		text = JSON.stringify(text)
	}
</script>

<main class="fathomChatContainer">
	<p>{@html text}</p>
</main>