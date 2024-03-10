<script lang="ts">
	import { goto } from '$app/navigation';
	import '../app.css';
	export let prerender = true;
	import '@fortawesome/fontawesome-free/css/all.min.css';
	import { page } from '$app/stores';
	import { createEventDispatcher } from 'svelte';
	let theme = 'light'; // Default theme

	$: currentPath = $page.url.pathname;

	function changePage(){
		if(currentPath == "/"){
			goto('/pattern');
		}
		else{
			goto('/');
		}
	}

	function openSidebar(){
		const myCustomEvent = new CustomEvent('sidebarEvents', {
			detail: {
				sidebarOpened: true
			}
		});
		document.dispatchEvent(myCustomEvent);
	}



</script>

<header>
	<div class="headerInner">
		<div class="logoWrapper">
			<button class="buttonCircled openSidebarBtn" on:click={openSidebar}><i class="fa-solid fa-bars"></i></button>
			<img src="./logo.png" alt="logo" class="logo" on:click={()=>goto('/')}/>
		</div>
		{#if currentPath === '/'}
		<button class="button" on:click={changePage}>Pattern</button>
		{:else if currentPath === '/pattern'}
		<button class="button" on:click={changePage}>Chat</button>
		{/if}
	</div>
</header>
<slot />
<style>
	header {
		position: fixed;
		width: 100%;
		top: 0;
		display: grid;
		grid-template-columns: 20% auto;
		min-height: 4rem;
		padding: 5px var(--page-horizontal-padding);
		z-index: 402;
		background-color: var(--color-white);
		border-bottom: 1px solid var(--color-sea-salt-gray);
		box-shadow: 0 4px 32px rgba(109,108,144,.12);
		display: flex;
		@media (max-width: 1080px) {
			padding: 5px 10px;
		}
	}

	.openSidebarBtn{
		display: none;
		@media (max-width: 1080px) {
			display: flex;
		}
	}

	.headerInner{
		margin: 0 auto;
		width: 100%;
		display: flex;
		justify-content: space-between;
		align-items: center;
	}

	.logo {
		height: 80px;
	}

	.logoWrapper{
		display: flex;
		justify-content: start;
		align-items: center;
	}

	.buttonCircled{
		margin-right: 10px;
	}

</style>
