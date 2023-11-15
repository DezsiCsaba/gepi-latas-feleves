import './assets/own.css'
import { createApp } from 'vue'
import App from './App.vue'

//VUETIFY STUFF
import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'
import '@mdi/font/css/materialdesignicons.css'
import myTheme from "./services/colorTheme";

const vuetify = createVuetify({
    icons: {
        defaultSet: 'mdi',
    },
    theme:
        {
        defaultTheme: 'light',
        // themes: {
        //     myTheme
        // }
    },
    components,
    directives,
})
const app = createApp(App)
app.use(vuetify)

app.mount('#app')
// createApp(App).mount('#app')
