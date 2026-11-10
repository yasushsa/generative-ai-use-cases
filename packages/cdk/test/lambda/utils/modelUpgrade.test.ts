import { readFileSync } from 'fs';
import { resolve } from 'path';
import { Model, UnrecordedMessage } from 'generative-ai-use-cases';
import {
  BEDROCK_SPEECH_TO_SPEECH_MODELS,
  BEDROCK_TEXT_MODELS,
  getSpeechToSpeechVoices,
  isSpeechToSpeechVoice,
  modelMetadata,
} from '@generative-ai-use-cases/common';
import { stackInputSchema } from '../../../lib/stack-input';

const context = JSON.parse(
  readFileSync(resolve(__dirname, '../../../cdk.json'), 'utf8')
).context;
const messages: UnrecordedMessage[] = [
  { role: 'system', content: 'Be helpful.' },
  { role: 'user', content: 'Hello' },
];
const originalEnv = process.env;
beforeEach(() => {
  jest.resetModules();
  process.env = {
    ...originalEnv,
    MODEL_IDS: JSON.stringify(
      context.modelIds.map((modelId: string) => ({
        modelId,
        region: context.modelRegion,
      }))
    ),
  };
});
afterAll(() => {
  process.env = originalEnv;
});

const createInput = async (model: Model) => {
  const { BEDROCK_TEXT_GEN_MODELS } = await import(
    '../../../lambda/utils/models'
  );
  const config = BEDROCK_TEXT_GEN_MODELS[model.modelId];
  return config.createConverseCommandInput(
    structuredClone(messages),
    '/rag',
    model,
    config.defaultParams,
    config.usecaseParams
  );
};

describe('latest-model deployment configuration', () => {
  it('accepts the configured deployment settings', () => {
    const parsed = stackInputSchema.parse(context);
    expect(parsed.modelRegion).toBe('us-east-1');
    expect(parsed.speechToSpeechModelIds).toEqual([
      'amazon.nova-2-5-sonic',
      'amazon.nova-2-sonic-v1:0',
    ]);
    expect(new Set(parsed.modelIds).size).toBe(parsed.modelIds.length);
  });

  it.each(context.modelIds as string[])(
    'has metadata and a backend handler for %s',
    async (modelId) => {
      const { BEDROCK_TEXT_GEN_MODELS } = await import(
        '../../../lambda/utils/models'
      );
      expect(BEDROCK_TEXT_MODELS).toContain(modelId);
      expect(modelMetadata[modelId].displayName).toBeTruthy();
      expect(BEDROCK_TEXT_GEN_MODELS[modelId]).toBeDefined();
      const input = await createInput({ type: 'bedrock', modelId });
      expect(input.modelId).toBe(modelId);
      expect(input.messages?.length).toBe(1);
    }
  );

  it.each([
    'global.anthropic.claude-sonnet-5-5',
    'global.anthropic.claude-haiku-5-5',
  ])(
    'omits sampling and requests summarized adaptive thinking for %s',
    async (modelId) => {
      const input = await createInput({
        type: 'bedrock',
        modelId,
        modelParameters: {
          reasoningConfig: {
            type: 'adaptive',
            budgetTokens: 0,
            effort: 'high',
          },
        },
      });
      expect(input.inferenceConfig?.temperature).toBeUndefined();
      expect(input.inferenceConfig?.topP).toBeUndefined();
      expect(input.additionalModelRequestFields).toEqual({
        thinking: { type: 'adaptive', display: 'summarized' },
        output_config: { effort: 'high' },
      });
    }
  );

  it('uses Kimi K3.1 Converse without Claude-specific fields or caching', async () => {
    const input = await createInput({
      type: 'bedrock',
      modelId: 'global.moonshotai.kimi-k3.1',
    });
    expect(input.additionalModelRequestFields).toBeUndefined();
    expect(input.system).toEqual([{ text: 'Be helpful.' }]);
    expect(input.messages).toEqual([
      { role: 'user', content: [{ text: 'Hello' }] },
    ]);
  });
});

describe('Nova Sonic voice metadata', () => {
  it.each(['amazon.nova-2-5-sonic', 'amazon.nova-2-sonic-v1:0'])(
    'offers supported, unique voices with the existing default for %s',
    (modelId) => {
      expect(BEDROCK_SPEECH_TO_SPEECH_MODELS).toContain(modelId);
      const voices = getSpeechToSpeechVoices(modelId);
      expect(voices[0].voiceId).toBe('tiffany');
      expect(voices[0].polyglot).toBe(true);
      expect(new Set(voices.map((v) => v.voiceId)).size).toBe(voices.length);
      expect(
        voices.every((v) => isSpeechToSpeechVoice(modelId, v.voiceId))
      ).toBe(true);
    }
  );
  it('does not accept a voice for an unknown model', () => {
    expect(getSpeechToSpeechVoices('unknown')).toEqual([]);
    expect(isSpeechToSpeechVoice('unknown', 'tiffany')).toBe(false);
  });
});
