#include "src/memllib/audio/AudioDriver.hpp"
#include "src/memllib/hardware/memlnaut/MEMLNaut.hpp"
#include <memory>
#include "src/memllib/interface/MIDIInOut.hpp"
#include "src/memllib/PicoDefs.hpp"

// Example apps and interfaces
#include "src/memllib/examples/IMLInterface.hpp"
#include "src/memllib/examples/SubtractiveSynthAudioApp.hpp"


// Global objects
std::shared_ptr<IMLInterface> interface;
std::shared_ptr<SubtractiveSynthAudioApp> audio_app;
std::shared_ptr<MIDIInOut> midi_interf;

// Inter-core communication
volatile bool core_0_ready = false;
volatile bool core_1_ready = false;
volatile bool serial_ready = false;
volatile bool interface_ready = false;


// We're only bound to the joystick inputs (x, y, rotate)
const size_t kN_InputParams = 3;


void bind_interface(std::shared_ptr<IMLInterface> interface)
{
    // Set up momentary switch callbacks
    MEMLNaut::Instance()->setMomA1Callback([interface] () {
        interface->Randomise();
    });
    MEMLNaut::Instance()->setMomA2Callback([interface] () {
        interface->ClearData();
    });

    // Set up toggle switch callbacks
    MEMLNaut::Instance()->setTogA1Callback([interface] (bool state) {
        interface->SetTrainingMode(state ? IMLInterface::TRAINING_MODE : IMLInterface::INFERENCE_MODE);
    });
    MEMLNaut::Instance()->setJoySWCallback([interface] (bool state) {
        interface->SaveInput(state ? IMLInterface::STORE_VALUE_MODE : IMLInterface::STORE_POSITION_MODE);
    });

    // Set up ADC callbacks
    MEMLNaut::Instance()->setJoyXCallback([interface] (float value) {
        interface->SetInput(0, value);
    });
    MEMLNaut::Instance()->setJoyYCallback([interface] (float value) {
        interface->SetInput(1, value);
    });
    MEMLNaut::Instance()->setJoyZCallback([interface] (float value) {
        interface->SetInput(2, value);
    });
    MEMLNaut::Instance()->setRVZ1Callback([interface] (float value) {
        // Scale value from 0-1 range to 1-3000
        value = 1.0f + (value * 2999.0f);
        interface->SetIterations(static_cast<size_t>(value));
    });

    // Set up loop callback
    MEMLNaut::Instance()->setLoopCallback([interface] () {
        interface->ProcessInput();
    });

    MEMLNaut::Instance()->setRVGain1Callback([interface] (float value) {
        AudioDriver::setDACVolume(value);
        Serial.println(value*4);
    });
}


void setup()
{
    Serial.begin(115200);
    while (!Serial) {}
    Serial.println("Serial initialised.");
    WRITE_VOLATILE(serial_ready, true);

    // Setup board
    MEMLNaut::Initialize();
    pinMode(33, OUTPUT);

    // Move MIDI setup after Serial is confirmed ready
    Serial.println("Initializing MIDI...");
    midi_interf = std::make_shared<MIDIInOut>();
    midi_interf->Setup(SubtractiveSynthAudioApp::kN_Params);
    midi_interf->SetMIDISendChannel(1);
    Serial.println("MIDI setup complete.");

    delay(100); // Allow Serial2 to stabilize

    // Setup interface with memory barrier protection
    {
        auto temp_interface = std::make_shared<IMLInterface>();
        MEMORY_BARRIER();
        temp_interface->setup(kN_InputParams, SubtractiveSynthAudioApp::kN_Params);
        MEMORY_BARRIER();
        temp_interface->SetMIDIInterface(midi_interf);
        MEMORY_BARRIER();
        interface = temp_interface;
        MEMORY_BARRIER();
    }
    WRITE_VOLATILE(interface_ready, true);

    // Bind interface after ensuring it's fully initialized
    bind_interface(interface);
    Serial.println("Bound interface to MEMLNaut.");

    WRITE_VOLATILE(core_0_ready, true);
    while (!READ_VOLATILE(core_1_ready)) {
        MEMORY_BARRIER();
        delay(1);
    }

    Serial.println("Finished initialising core 0.");
}

void loop()
{
    MEMORY_BARRIER();
    MEMLNaut::Instance()->loop();
    midi_interf->Poll();  // Only poll MIDI in core 0
    MEMORY_BARRIER();
    static int blip_counter = 0;
    if (blip_counter++ > 100) {
        blip_counter = 0;
        Serial.println(".");
        // Blink LED
        digitalWrite(33, HIGH);
    } else {
        // Un-blink LED
        digitalWrite(33, LOW);
    }
    delay(10); // Add a small delay to avoid flooding the serial output
}

void setup1()
{
    while (!READ_VOLATILE(serial_ready)) {
        MEMORY_BARRIER();
        delay(1);
    }

    while (!READ_VOLATILE(interface_ready)) {
        MEMORY_BARRIER();
        delay(1);
    }

    // Create audio app with memory barrier protection
    {
        auto temp_audio_app = std::make_shared<SubtractiveSynthAudioApp>();
        temp_audio_app->Setup(AudioDriver::GetSampleRate(), interface);
        MEMORY_BARRIER();
        audio_app = temp_audio_app;
        MEMORY_BARRIER();
    }

    // Start audio driver
    AudioDriver::Setup();

    WRITE_VOLATILE(core_1_ready, true);
    while (!READ_VOLATILE(core_0_ready)) {
        MEMORY_BARRIER();
        delay(1);
    }

    Serial.println("Finished initialising core 1.");
}

void loop1()
{
    // Audio app parameter processing loop
    audio_app->loop();
}
