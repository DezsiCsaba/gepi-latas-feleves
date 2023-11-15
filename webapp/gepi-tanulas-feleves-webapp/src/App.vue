<template>
  <div>
    <v-layout>
      <v-app-bar
          color="primary"
      >
        <v-app-bar-title>Féléves Feladat Demo APP</v-app-bar-title>
      </v-app-bar>

      <v-main>
        <div style="display: flex">
          <v-card style="width: max-content">
            <v-card-title>Inputs of the model</v-card-title>
            <v-form style="margin: 15px">
              <v-autocomplete
                  v-model="this.selectedPlatform"
                  variant="underlined"
                  density="compact"
                  base-color="primary"
                  item-color="background"
                  label="Platform"
                  :items="platformList"
              ></v-autocomplete>
              <v-autocomplete
                  v-model="this.selectedGenre"
                  variant="underlined"
                  density="compact"
                  base-color="primary"
                  item-color="background"
                  label="Genre"
                  :items="genreList"
              ></v-autocomplete>

              Editors's choice Y/N:
              <v-switch
              ></v-switch>

              Date of release:
              <v-date-picker></v-date-picker>
            </v-form>

            <v-btn
                @click="setUpPredictData(this.selectedPlatform, this.selectedGenre)"
            >Predict</v-btn>
          </v-card>

          <v-card style="width: max-content">
            <v-card-title>The model's prediction</v-card-title>
          </v-card>
        </div>
      </v-main>
    </v-layout>
  </div>
</template>

<script setup>
  import platformNames from '../../../service/platformName.json'
  import genreNames from '../../../service/genreName.json'
  import editorsChoice from '../../../service/editorsChoice.json'
  import {onBeforeMount} from "vue";
  import {tr} from "vuetify/locale";

  //#region Lists
  function setLists(){
    Object.keys(platformNames).forEach((key) => {
      platformIndexes.push(key)
      platformList.push(platformNames[key])
    })
    Object.keys(genreNames).forEach((key) => {
      genreIndexes.push(key)
      genreList.push(genreNames[key])
    })
    Object.keys(editorsChoice).forEach((key) => {
      editorsChoiceIndexes.push(key)
    })
  }
  function getMatchingIndex(lookup, input){
    let keyVal = null
    Object.keys(lookup).forEach((key)=>{
      if (lookup[key] === input) {
        keyVal = key
        return
      }
    })
    return keyVal
  }
  function setUpPredictData(selectedPlatform, selectedGenre){
    let platformI = getMatchingIndex(platformNames, selectedPlatform)
    let genreI = getMatchingIndex(genreNames, selectedGenre)

    console.log(`${platformI}: ${selectedPlatform}, ${genreI}: ${selectedGenre}`)
  }

  const platformList = []
  const platformIndexes = []
  const genreList = []
  const genreIndexes = []
  const editorsChoiceOptions = [true, false]
  const editorsChoiceIndexes = []
  //#endregion

  onBeforeMount(()=> {
    setLists()
  })
</script>

<script>
export default {
  data(){return{
    selectedPlatform: 'PlayStation Vita',
    selectedGenre: 'Platformer',
    selectedEditorsChoice: ''
  }}
}
</script>

<style scoped>
</style>
