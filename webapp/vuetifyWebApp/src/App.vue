<template>
  <v-app>
    <v-main>
      <v-layout>
        <v-app-bar
          color="primary"
        >
          <v-app-bar-title>Féléves Feladat Demo APP</v-app-bar-title>
        </v-app-bar>
        <v-main>
          <div style="display: flex; flex-direction: row; align-items: flex-start">
            <div style="width: 100%; padding: 20px 300px 20px 300px; margin: 10px; display: flex; align-items: start; flex-direction: column">

              <v-card style="width: 100%; height: 100%; margin: 10px; padding: 20px; display: flex; justify-items: start; flex-direction: column">
                <v-card-title>Inputs of the model</v-card-title>
                <v-timeline
                  align="start"
                  line-inset="10px"
                  side="end"
                  style="justify-content: left; grid-template-columns: .01fr .1fr 1fr; width: 100%"
                >
                  <v-timeline-item dot-color="primary" width="100%">
                    <v-card-title>Platform and Genre</v-card-title>
                    <v-autocomplete
                      v-model="selectedPlatform"
                      variant="underlined"
                      density="compact"
                      base-color="primary"
                      label="Platform"
                      :items="platformList"
                      :hide-selected="true"
                      prepend-inner-icon="mdi-controller-classic-outline"
                    ></v-autocomplete>
                    <v-autocomplete
                      v-model="selectedGenre"
                      variant="underlined"
                      density="compact"
                      base-color="primary"
                      label="Genre"
                      :items="genreList"
                      :hide-selected="true"
                      prepend-inner-icon="mdi-script-text-outline"
                    ></v-autocomplete>
                  </v-timeline-item>

                  <v-timeline-item dot-color="primary" width="100%">
                    <v-card-title>Editor's choice (Y/N)</v-card-title>
                    <v-switch
                      v-model="selectedEditorsChoice"
                      :inset="true"
                      base-color="primary"
                    ></v-switch>
                  </v-timeline-item>

                  <v-timeline-item dot-color="primary" width="100%">
                    <v-card-title>Date of release:</v-card-title>

                    <v-menu
                      :close-on-content-click="false"
                    >
                      <template v-slot:activator="{props}">
                        <v-text-field
                          v-bind="props"
                          label="Release date"
                          prepend-inner-icon="mdi-calendar-range"
                          variant="underlined"
                          base-color="primary"
                          :model-value="formatedDate"
                        ></v-text-field>
                      </template>
                      <v-date-picker
                        v-model="releaseDate"
                        color="secondary"
                      ></v-date-picker>
                    </v-menu>


                  </v-timeline-item>

                </v-timeline>
              </v-card>

              <v-card style="width: 100%; padding: 20px; margin: 10px">
                <v-card-title>The model's prediction</v-card-title>
                <v-form>
                  <v-hover>
                    <template v-slot:default="{ isHovering, props }">
                      <v-btn
                        v-bind="props"
                        :color="isHovering ? 'secondary' : 'primary'"
                        style="cursor: pointer"
                        @click="runTheModel"
                      >Prediction test</v-btn>
                    </template>
                  </v-hover>

                  <v-card-text>The predicted value:</v-card-text>
                  <v-text-field
                    :readonly="true"
                    v-model="prediction"
                  ></v-text-field>
                </v-form>
              </v-card>
            </div>


          </div>
        </v-main>
      </v-layout>
    </v-main>
  </v-app>
</template>

<script setup>
  const showInnerLogs = false
  const showFunctionLogs = false
  const showPredVal = true

  import platformNames from '../../../service/platformName.json'
  import genreNames from '../../../service/genreName.json'
  // import editorsChoice from '../../../service/editorsChoice.json'
  import scalerImport from '../../../service/sk_scaler.json'

  import {computed, onBeforeMount, ref} from "vue";
  import {InferenceSession, Tensor} from "onnxruntime-web";
  import * as tf from '@tensorflow/tfjs'
  import * as sk from 'scikitjs'
  import {format} from 'date-fns'

  //#region props
  const platformList = []
  const platformIndexes = []
  const genreList = []
  const genreIndexes = []

  let releaseDate = ref(new Date())
  let formatedDate = computed({
    get() {
      return format(releaseDate.value, 'YYY MMM d')
    },
    set(newValue) {
      releaseDate.value = newValue
    }
  })

  let selectedPlatform = ref('PlayStation 4')
  let selectedGenre = ref('Adventure, RPG')
  let selectedEditorsChoice = ref(true)
  let prediction = ref('NAN')


  let platformInt
  let genreInt
  let editorsChoiceInt

  defineExpose({
    selectedPlatform, selectedGenre, selectedEditorsChoice,

    platformInt, genreInt, editorsChoiceInt
  })
  //#endregion

  //#region getDataFromFrontEnd
  function setLists(){
    Object.keys(platformNames).forEach((key) => {
      platformIndexes.push(key)
      platformList.push(platformNames[key])
    })
    Object.keys(genreNames).forEach((key) => {
      genreIndexes.push(key)
      genreList.push(genreNames[key])
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
  function setUpPredictData() {
    console.clear()
    logFunctinStart('The input from the front:')

    platformInt = getMatchingIndex(platformNames, selectedPlatform.value)
    genreInt = getMatchingIndex(genreNames, selectedGenre.value)
    editorsChoiceInt = selectedEditorsChoice.value === true ?10 : 20

    logInsideOfFunction(`platform = ${platformInt},\n\tgenre = ${genreInt},\n\teditor's choice = ${editorsChoiceInt}`)
  }
  //#endregion

  async function preProcessData() {
    logFunctinStart('Preprocessing data')
    // const standardScaler = createScaler()

    let scaler = new sk.StandardScaler()

    //let x_test = [[1, 1, 10, 2010, 1, 1]]
    let y = releaseDate.value.getFullYear()
    let m = releaseDate.value.getMonth()+1
    let d = releaseDate.value.getDate()
    let x = [[+platformInt, +genreInt, +editorsChoiceInt, y, m, d]]

    scaler.withMean = scalerImport.with_mean
    scaler.withStd = scalerImport.with_std
    scaler.copy = scalerImport.copy
    scaler.featureNamesIn = scalerImport.feature_names_in_
    scaler.nFeaturesIn = scalerImport.n_features_in_
    scaler.nSamplesSeen = scalerImport.n_samples_seen_
    scaler.mean = tf.tensor(scalerImport.mean_)
    scaler.scale = tf.tensor(scalerImport.scale_)

    //let transformed = await scaler.transform(x_test).data()
    let transformed = await scaler.transform(x).data()

    console.log({
      input_before_scaling: x,
      input_after_scaling: transformed
    })

    return new Tensor('float32', Float32Array.from(transformed), [1, 6])
  }

  async function runTheModel(){
    setUpPredictData()
    const x_test = await preProcessData()
    prediction.value = await pred(x_test)
  }

  async function pred(inputData){
    logFunctinStart('Prediction with input data')
    try{
      const session = await InferenceSession.create(
        './public/onnx_model.onnx',
        {
          executionProviders: ["webgl"]
        }
      )

      console.log({inputData: inputData})
      const outputMap = await session.run({'onnx::Gemm_0': inputData})

      let y_pred = null
      Object.keys(outputMap).forEach((key) => y_pred = outputMap[key].data[0])

      logYpred({inputs: inputData.data, y_pred: y_pred})
      logYpred(`${y_pred.toString().slice(0, 7)}`)
      return `${y_pred.toString().slice(0, 7)}`
    }catch (err){
      console.error(err.stack)
    }
  }

  async function test(){
    logFunctinStart('test prediction with tensor.ones')
    const session = await InferenceSession.create(
      './public/onnx_model.onnx',
      {
        executionProviders: ["webgl"]
      }
    )

    const inputs = new Tensor('float32', Float32Array.from([1, 1, 1, 1, 1, 1]), [1,6])
    const outputMap = await session.run({'onnx::Gemm_0': inputs})

    let y_pred = null
    Object.keys(outputMap).forEach((key) => y_pred = outputMap[key].data[0])

    logYpred(`${y_pred.toString().slice(0, 7)}`)
  }

  function createArrayFromJSON(jsonObj){
    let out = []
    Object.keys(jsonObj).forEach((key) => {
      out.push(jsonObj[key])
    })
    return out
  }

  //#region loggers
  function logFunctinStart(input){
    showFunctionLogs && console.log('>>>', input)
  }
  function logInsideOfFunction(input){
    showInnerLogs && console.log('\t', input)
  }
  function logYpred(input){
    showPredVal && console.log('\t< pred >', input)
  }
  //#endregion

  onBeforeMount(()=> {
    setLists()
    sk.setBackend(tf)
  })
</script>
