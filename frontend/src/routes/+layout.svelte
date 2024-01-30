<script lang="ts">
	import '../app.css';
	export const prerender = true;
	import '@fortawesome/fontawesome-free/css/all.min.css'
</script>

<header>
	<div class="headerInner">
		<img src="./logo.png" alt="logo" class="logo"/>
	</div>
</header>
<slot />
<style>
	header {
		position: fixed;
		width: 100%;
		top: 0;
		display: grid;
		grid-template-columns: 20% 80%;
		min-height: 4rem;
		padding: 5px var(--page-horizontal-padding);

		top: 0;
		z-index: 402;
		background-color: var(--color-white);
		border-bottom: 1px solid var(--color-sea-salt-gray);
		-webkit-box-shadow: 0 4px 32px rgba(109,108,144,.12);
		box-shadow: 0 4px 32px rgba(109,108,144,.12);

	}

	.headerInner{
		max-width: var(--max-header-width);
		margin: 0 auto;
		width: 100%;
	}

	.logo {
		height: 80px;
		max-width: var(--max-header-width);
	}
</style>
