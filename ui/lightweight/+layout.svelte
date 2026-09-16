<script lang="ts">
  import '../app.css';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { ModeWatcher } from 'mode-watcher';
  let { children } = $props();
</script>
<ModeWatcher defaultMode="light" />
<Tooltip.Provider>{@render children()}</Tooltip.Provider>
