import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()]
});

/* import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig, loadEnv } from 'vite';

/** @type {import('vite').UserConfig} */
// @ts-ignore
/* export default ({ mode }) => {
    // Extends 'process.env.*' with VITE_*-variables from '.env.(mode=production|development)'
    process.env = {...process.env, ...loadEnv(mode, process.cwd())};
    return defineConfig({
        plugins: [sveltekit()]
    }); 
}; */
