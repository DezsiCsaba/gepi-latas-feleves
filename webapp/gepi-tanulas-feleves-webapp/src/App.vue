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
                  v-model="selectedPlatform"
                  variant="underlined"
                  density="compact"
                  base-color="primary"
                  item-color="background"
                  label="Platform"
                  :items="platformList"
              ></v-autocomplete>
              <v-autocomplete
                  v-model="selectedGenre"
                  variant="underlined"
                  density="compact"
                  base-color="primary"
                  item-color="background"
                  label="Genre"
                  :items="genreList"
              ></v-autocomplete>

              <v-switch
                  v-model="selectedEditorsChoice"
                  :inset="true"
                  label="Editor's choice (Y/N)"
              ></v-switch>

              Date of release:
              <v-date-picker></v-date-picker>
            </v-form>
          </v-card>

          <v-card style="width: max-content">
            <v-card-title>The model's prediction</v-card-title>

            <v-btn
                @click="setUpPredictData()"
            >Prediction test</v-btn>
          </v-card>
        </div>
      </v-main>
    </v-layout>
  </div>
</template>

<script setup>
  import platformNames from '../../../service/platformName.json'
  import genreNames from '../../../service/genreName.json'
  // import editorsChoice from '../../../service/editorsChoice.json'
  import {onBeforeMount, ref} from "vue";
  import {InferenceSession, Tensor} from "onnxruntime-web";

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
  async function setUpPredictData() {
    console.clear()
    console.log(selectedEditorsChoice.value)

    platformInt = getMatchingIndex(platformNames, selectedPlatform.value)
    genreInt = getMatchingIndex(genreNames, selectedGenre.value)
    editorsChoiceInt = selectedEditorsChoice.value === true ?10 : 20

    console.log(`platform = ${platformInt},\ngenre = ${genreInt},\neditor's choice = ${editorsChoiceInt}`)

    await this.test()
  }

  const platformList = []
  const platformIndexes = []
  const genreList = []
  const genreIndexes = []
  //#endregion

  let selectedPlatform = ref('PlayStation 4')
  let selectedGenre = ref('Adventure, RPG')
  let selectedEditorsChoice = ref(true)

  let platformInt
  let genreInt
  let editorsChoiceInt

  defineExpose({
    selectedPlatform, selectedGenre, selectedEditorsChoice,

    platformInt, genreInt, editorsChoiceInt
  })

  async function test(){
    // console.log('test')
    // try{
    //   const session = await InferenceSession.create('../../../onnx_model.onnx',{
    //     executionProviders: ["webgl"],
    //   })
    //   console.log('model imported')
    //
    //   const inputs = [
    //     new Tensor(new Float32Array([platformInt, genreInt, editorsChoiceInt, 2012.0, 9.0, 12.0]))
    //   ]
    //
    //   console.log({inputTensor: inputs})
    //   const outputMap = await session.run(inputs)
    //   const outputTensor = outputMap.values().next().value
    //
    //   console.log(outputTensor)
    // }catch (err){
    //   console.error(err.stack)
    // }
    const session = await InferenceSession.create(
        './public/onnx_model.onnx',
        {
          executionProviders: ["webgl"]
        }
    )
    console.log('model imported')

    const inputs = new Tensor('float32', Float32Array.from([1, 1, 1, 1, 1, 1]), [1,6])
    console.log(inputs)

    const outputMap = await session.run({'onnx::Gemm_0': inputs})

    let y_pred = null
    Object.keys(outputMap).forEach((key) => y_pred = outputMap[key].data[0])
    console.log(y_pred)
  }



  onBeforeMount(()=> {
    setLists()
  })
</script>

<style scoped>
</style>
