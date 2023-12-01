<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import {Shadow} from 'svelte-loading-spinners';

	const dispatch = createEventDispatcher();

	let value = '';
	let placeholder = 'Type your prompt here';
	let heightModifier = 3.2;
	let textarea: HTMLTextAreaElement;
	let loading = false;

	export function toggleLoading() {
		loading = !loading
		console.log('toggling loading to ', loading);
	}

	function submitPrompt() {
		if (value !== '') {
			dispatch('submit', value);
		}
	}

	function detectEnterPress(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			submitPrompt();
			textarea.value = '';
			heightModifier = 3.2;
			textarea.focus();
		} else if (e.key === 'Enter' && e.shiftKey) {
			heightModifier += 1;
		}
	}
</script>

<main>
	{#if loading}
	<Shadow size="30" color="#FFFFFF" unit="px" duration="1s" />
	{/if}
	<div>
		<!-- svelte-ignore a11y-autofocus -->
		<textarea
			bind:this={textarea}
			bind:value
			autofocus
			{placeholder}
			on:keypress={detectEnterPress}
			style="--heightModifier: {heightModifier}"
		/>
		<button on:click={submitPrompt}>
			<img src="/submit.svg" alt="submission arrow" />
		</button>
	</div>
</main>

<style>
	main {
		position: sticky;
		background-image: linear-gradient(
			180deg,
			rgba(0, 0, 0, 0) 0%,
			var(--background-dark) 50%,
			var(--background-dark) 100%
		);
		width: 100%;
		margin: 0 auto;
		bottom: 0;
		left: 0;
		display: grid;
		place-items: center;
		min-height: 12rem;
		overscroll-behavior: none;
	}
	div {
		place-self: center;
		width: 70%;
		border: none;
		border-radius: 0.5rem;
		padding: 0.5rem 0.8rem;
		background-color: var(--accent-light);
		color: var(--accent-dark);
		display: flex;
		align-items: center;
		justify-content: center;
		box-shadow: 0 0 2rem rgba(255, 255, 255, 0.2);
	}
	textarea {
		width: 100%;
		height: calc(var(--heightModifier) * 1rem);
		padding-top: 1rem;
		font-size: 1rem;
		background-color: transparent;
		border: none;
		resize: none;
		display: grid;
		place-items: center;
	}
	textarea:focus {
		outline: none;
	}
	button {
		background-color: rgba(255, 255, 255, 0.5);
		padding: 0.2rem;
		border-radius: 0.2rem;
		width: 2.5rem;
		height: 2.5rem;
		border: none;
		place-self: center end;
	}
	button:hover {
		background-color: rgba(255, 255, 255, 0.8);
	}
	button:active {
		background-color: rgba(0, 0, 0, 0.5);
	}

	img {
		width: 100%;
		height: 100%;
	}
</style>
