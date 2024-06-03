<script lang="ts">
	export let responseText: string;
	export let videoUrl: string;


</script>

<main class="fathomChatContainer">
		<p>{responseText}</p>
		<br/>
		<video controls="true" src="{videoUrl}"/>

</main>

<style>
	video{
		max-width: min(100%, 500px);
		max-height: 500px;
		width: auto;
		height: auto;
		padding: 10px 0px;
	}
</style>