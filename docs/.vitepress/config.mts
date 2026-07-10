import { defineConfig } from 'vitepress';

const guideSidebar = [
  {
    text: 'Guide',
    items: [
      { text: 'Getting Started', link: '/guide/quick-start' },
    ],
  },
];

const usageSidebar = [
  {
    text: 'Usage',
    items: [
      { text: 'Data and Outputs', link: '/usage/data-structure' },
      { text: 'Configuration and Tuning', link: '/usage/parameter-tuning' },
    ],
  },
];

const workflowsSidebar = [
  {
    text: 'Workflows',
    items: [
      { text: 'MuJoCo Warp (MJWP)', link: '/workflows/workflow-mjwp' },
      { text: 'Optional Backends', link: '/workflows/optional-backends' },
    ],
  },
];

const developmentSidebar = [
  {
    text: 'Development',
    items: [
      { text: 'Add a Dataset', link: '/development/add-dataset' },
      { text: 'Add a Robot', link: '/development/add-robot' },
      { text: 'Add a Simulator', link: '/development/add-simulator' },
    ],
  },
];

export default defineConfig({
  lang: 'en-US',
  title: 'SPIDER',
  description: 'Scalable Physics-Informed Dexterous Retargeting',
  base: '/spider/',
  head: [['link', { rel: 'icon', href: '/favicon.ico' }]],
  themeConfig: {
    nav: [
      { text: 'Guide', link: '/guide/quick-start' },
      { text: 'Usage', link: '/usage/data-structure' },
      { text: 'Workflows', link: '/workflows/workflow-mjwp' },
      { text: 'Development', link: '/development/add-dataset' },
      { text: 'GitHub', link: 'https://github.com/facebookresearch/spider' },
    ],
    sidebar: {
      '/guide/': guideSidebar,
      '/usage/': usageSidebar,
      '/workflows/': workflowsSidebar,
      '/development/': developmentSidebar,
    },
    socialLinks: [{ icon: 'github', link: 'https://github.com/facebookresearch/spider' }],
  },
});
